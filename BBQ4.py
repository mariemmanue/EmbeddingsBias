#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BBQ.py — RQ4: Entity-Level Effects (Positional Bias)
----------------------------------------------------

Measure whether swapping the order of two entities (A-B vs B-A) in the context
changes cosine similarity to a *negative* question.

Pipeline: 
1) Select rows (optionally only negative polarity; ambig/disambig filter).
2) Extract exactly two entities from answer_info.
3) Detect which entity appears first → label as "A-B" or "B-A".
4) If both orders exist in the dataset for the same (pair_id, context_condition), use them.
   Otherwise attempt a programmatic swap; if that fails and RITS is provided, try an LLM swap.
5) Embed question + contexts (Granite embeddings), compute cos sims, Δcos = (A-B) − (B-A).
6) Stats (overall, by category), positional split (whether first entity is stereotyped).
7) Save CSVs and a bar plot.
"""

from __future__ import annotations

import os, re, json, argparse, hashlib, warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import ttest_1samp, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import ast
import math


# -----------------------------
# Config & light dependencies
# -----------------------------

DET_WORDS = {"a","an","the","one"}
HUMAN_NOUNS = {
    "person","man","woman","boy","girl","guy","men","women","people","friend",
    "classmate","student","acquaintance","sibling","brother","sister","roommate","babysitter",
}

# a/an heuristics
VOWEL_SOUND_EXCEPTIONS_AN = {"hour","honest","heir","honor"}
VOWEL_LETTER_BUT_CONSONANT_SOUND = {"university","unicorn","european","unit","user","one","once"}

# light category map for later positional flags
_CAT_MAP = {
    "gender_identity": "GENDER_IDENTITY",
    "race_x_gender": "RACE_X_GENDER",
    "race_x_ses": "RACE_X_SES",
    "race_ethnicity": "RACE_ETHNICITY",
    "nationality": "NATIONALITY",
    "physical_appearance": "PHYSICAL_APPEARANCE",
    "sexual_orientation": "SEXUAL_ORIENTATION",
    "religion": "RELIGION",
    "ses": "SES",
    "age": "AGE",
    "disability_status": "DISABILITY_STATUS",
}

def canon_cat(cat: str) -> str:
    if not isinstance(cat, str): return ""
    key = cat.strip().lower().replace(" ","_")
    return _CAT_MAP.get(key, cat.upper())

# -----------------------------
# Light text utilities
# -----------------------------
def _is_nanlike(x):
    return (
        x is None
        or (isinstance(x, float) and math.isnan(x))
        or (isinstance(x, str) and x.strip() == "")
    )

def _py_like_to_python(s: str) -> str:
    """
    Turn a string like:
      "{'ans0': array(['The grandfather','grandfather','old'], dtype=object), ...}"
    into something ast.literal_eval can read:
      "{'ans0': ('The grandfather','grandfather','old'), ...}"
    """
    if not isinstance(s, str):
        return s
    t = s.strip()
    # Replace array([...], dtype=object) -> ([...])
    t = re.sub(r"\barray\(", "(", t)
    t = re.sub(r",\s*dtype\s*=\s*object\)", ")", t)
    # Normalize fancy quotes sometimes seen in logs
    t = (t.replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("\u2019", "'"))
    return t

def _coerce_answer_info(ai_raw):
    """
    Return a dict when possible, else None.
    Supports:
      - real dict
      - stringified Python dict with NumPy array(...) repr
    """
    if isinstance(ai_raw, dict):
        return ai_raw
    if isinstance(ai_raw, str):
        t = _py_like_to_python(ai_raw)
        try:
            val = ast.literal_eval(t)
            return val if isinstance(val, dict) else None
        except Exception:
            return None
    return None


def _json_load_maybe(x):
    if isinstance(x, (dict, list)): 
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                return x  # leave as-is
    return x

import math
import numpy as np

def _to_list(x):
    """Coerce stereotyped_groups into a plain Python list of strings."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple, set)):
        return list(x)
    # scalars (incl. single string)
    return [x]


def clean(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.strip())

def strip_dets(s: str) -> str:
    toks = clean(s).split()
    while toks and toks[0].lower() in DET_WORDS: toks = toks[1:]
    while toks and toks[-1].lower() in DET_WORDS: toks = toks[:-1]
    return " ".join(toks)

def canon_tokens(s: str) -> List[str]:
    s = strip_dets(s).strip(" .,:;!?\"'()[]{}")
    toks = [t.lower() for t in s.split() if t]
    while toks and toks[0] in DET_WORDS: toks = toks[1:]
    return toks

def fix_indef_articles(s: str) -> str:
    """Heuristic a/an fixer with common exceptions."""
    if not isinstance(s, str): return s

    def choose(word: str) -> str:
        wl = word.lower()
        if wl in VOWEL_SOUND_EXCEPTIONS_AN: return "an"
        if wl in VOWEL_LETTER_BUT_CONSONANT_SOUND: return "a"
        return "an" if wl and wl[0] in "aeiou" else "a"

    def repl(m: re.Match) -> str:
        orig = m.group(1)
        nxt  = m.group(2)
        good = choose(nxt)
        if orig[0].isupper():
            good = good.capitalize()
        return f"{good} {nxt}"

    return re.sub(r"\b(a|an)\s+([A-Za-z][A-Za-z\-']*)\b", repl, s, flags=re.IGNORECASE)

# -----------------------------
# Entity extraction / matching
# -----------------------------

# --- helpers (near your other small helpers) ---
_UNKNOWN_PATTERNS = {
    "unknown",
    "can't be determined",
    "cannot be determined",
    "not enough information",
    "insufficient information",
    "cannot answer",
    "no answer",
    "undetermined",
    "n/a", "can't be determined", "not enough info", "unknown",
    "can't answer", "cannot answer", "cannot be determined",
    "not known", "undetermined"
}

def _norm_txt(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().lower()

def _is_unknown_option(name: str, label: str) -> bool:
    t = f"{_norm_txt(name)} {_norm_txt(label)}"
    # exact-ish hits first
    if _norm_txt(label) in {"unknown", "none of the above"}:
        return True
    # contains common unknown phrasings
    return any(pat in t for pat in _UNKNOWN_PATTERNS)

def _coerce_pair(seq) -> tuple[str, str] | None:
    """Return (name,label) from ndarray/list/tuple/dict; else None."""
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    if isinstance(seq, (list, tuple)) and len(seq) >= 2:
        return str(seq[0] or ""), str(seq[1] or "")
    if isinstance(seq, dict):
        return str(seq.get("name") or ""), str(seq.get("label") or "")
    return None

# --- main extractor (drop-in replacement) ---
def extract_two_entities_from_row(row: pd.Series) -> Optional[List[Tuple[str, str]]]:
    """
    Build two entity options from answer_info WITHOUT assuming which index is Unknown.
    Strategy:
      1) Collect all options present (ans0/ans1/ans2/...).
      2) Clean (name,label) and mark unknown by content.
      3) Choose the first two NON-UNKNOWN; if only one exists, backfill with one UNKNOWN.
    """
    ai = row.get("answer_info")
    if not isinstance(ai, dict):
        # keep fallbacks if you have them; most of your rows are dicts per the logs
        return None

    # 1) Gather candidates in a stable order: ans0, ans1, ans2, then other keys sorted.
    ordered_keys = [k for k in ("ans0", "ans1", "ans2") if k in ai] + \
                   [k for k in sorted(ai.keys()) if k not in {"ans0", "ans1", "ans2"}]

    cands: list[dict] = []
    for k in ordered_keys:
        pair = _coerce_pair(ai.get(k))
        if pair is None:
            continue
        name_raw, label_raw = pair
        name = strip_dets(clean(name_raw))
        label = clean(label_raw)
        if not (name or label):
            continue
        cands.append({
            "key": k,
            "name": name,
            "label": label,
            "is_unknown": _is_unknown_option(name, label),
        })

    if not cands:
        # optional fallback to flat columns, if you keep that pathway
        a0n = row.get("ans0_name"); a0l = row.get("ans0_label")
        a1n = row.get("ans1_name"); a1l = row.get("ans1_label")
        out = []
        if (a0n or a0l): out.append((strip_dets(clean(a0n or "")), clean(a0l or "")))
        if (a1n or a1l): out.append((strip_dets(clean(a1n or "")), clean(a1l or "")))
        return out[:2] if out else None

    # 2) Choose two
    non_unknown = [c for c in cands if not c["is_unknown"]]
    unknown     = [c for c in cands if c["is_unknown"]]

    chosen: list[tuple[str,str]] = []
    for c in non_unknown:
        if len(chosen) < 2:
            chosen.append((c["name"], c["label"]))
    if len(chosen) < 2:
        for c in unknown:
            if len(chosen) < 2:
                chosen.append((c["name"], c["label"]))

    return chosen[:2] if len(chosen) >= 2 else None


def build_name_regex(name: str) -> re.Pattern:
    toks = [t for t in canon_tokens(name) if t not in DET_WORDS]
    if not toks:  # fallback never-matches neutral, caller will try label
        return re.compile(r"$a", re.I)
    # if last token is a human noun, allow optional noun backoff
    if toks[-1] in HUMAN_NOUNS:
        core = r"\s+".join(map(re.escape, toks[:-1])) if len(toks) > 1 else ""
        if core:
            pat = rf"\b{core}\b(?:\s+(?:{'|'.join(map(re.escape,HUMAN_NOUNS))}))?"
        else:
            pat = rf"\b(?:{'|'.join(map(re.escape,HUMAN_NOUNS))})\b"
    else:
        pat = r"\b" + r"\s+".join(map(re.escape, toks)) + r"\b"
    return re.compile(pat, re.I)

def build_label_regex(label: str) -> Optional[re.Pattern]:
    if not label: return None
    return re.compile(rf"\b{re.escape(label.strip())}\b", re.I)

def find_span(context: str, ent: Tuple[str,str], category: Optional[str]) -> Optional[re.Match]:
    if not isinstance(context, str): 
        return None
    name, label = ent
    # try NAME (regex)
    m = build_name_regex(name).search(context)
    if m: 
        return m
    # try LABEL (regex)
    lab = build_label_regex(label)
    if lab:
        m = lab.search(context)
        if m:
            return m
    # FINAL FALLBACK: fuzzy substring of either name or label (first hit wins)
    for needle in (name, label):
        s = (needle or "").strip()
        if not s:
            continue
        i = context.lower().find(s.lower())
        if i >= 0:
            # synthesize a zero-width-like match over [i, i+len(s)]
            class _M:
                def __init__(self, s, e): self._s, self._e = s, e
                def start(self): return self._s
                def end(self):   return self._e
            return _M(i, i+len(s))
    return None


def both_in_context(context: str, e1: Tuple[str,str], e2: Tuple[str,str], category: Optional[str]=None) -> bool:
    return (find_span(context, e1, category) is not None and
            find_span(context, e2, category) is not None)

def entity_order(context: str, e1: Tuple[str,str], e2: Tuple[str,str], category: Optional[str]=None) -> Optional[str]:
    m1 = find_span(context, e1, category)
    m2 = find_span(context, e2, category)
    if not (m1 and m2) or m1.start() == m2.start(): return None
    return "A-B" if m1.start() < m2.start() else "B-A"

# -----------------------------
# Swapping
# -----------------------------

def expand_span_to_np(text: str, m: re.Match) -> tuple[int,int]:
    """Expand to include preceding DET/human-noun and trailing 'of mine'."""
    start, end = m.start(), m.end()
    tail = re.match(r"\s+of mine\b", text[end:], flags=re.I)
    if tail: end += tail.end()
    pre = text[:start]
    w = re.search(r"\b([A-Za-z][A-Za-z\-']*)\b\s*$", pre)
    if w and w.group(1).lower() in HUMAN_NOUNS:
        nstart = w.start(1)
        det = re.search(r"\b(?:a|an|the|one)\b\s*$", text[:nstart], flags=re.I)
        start = det.start() if det else nstart
    else:
        det = re.search(r"\b(?:a|an|the|one)\b\s*$", pre, flags=re.I)
        if det: start = det.start()
    return start, end

def programmatic_swap(context: str, e1: Tuple[str,str], e2: Tuple[str,str], category: Optional[str]=None) -> Optional[str]:
    """Swap first mentions by replacing entire NPs for e1/e2."""
    if not isinstance(context, str): return None
    m1 = find_span(context, e1, category)
    m2 = find_span(context, e2, category)
    if not (m1 and m2) or m1.start() == m2.start(): return None
    # ensure m1 is earlier
    if m1.start() > m2.start(): (e1, e2), (m1, m2) = (e2, e1), (m2, m1)
    s1, e1_ = expand_span_to_np(context, m1)
    s2, e2_ = expand_span_to_np(context, m2)
    first, mid, last = context[:s1], context[e1_:s2], context[e2_:]
    span1, span2 = context[s1:e1_], context[s2:e2_]
    swapped = first + span2 + mid + span1 + last
    return fix_indef_articles(swapped)

def verify_flipped(original: str, swapped: str, e1: Tuple[str,str], e2: Tuple[str,str], category: Optional[str]=None) -> bool:
    """Minimal: order must flip and both entities still present."""
    o1 = entity_order(original, e1, e2, category)
    o2 = entity_order(swapped,  e1, e2, category)
    return (o1 in {"A-B","B-A"}) and (o2 in {"A-B","B-A"}) and (o1 != o2)

# -----------------------------
# IDs & pairing
# -----------------------------

def stable_norm_key(s: str) -> str:
    return hashlib.sha1(clean(s).encode("utf-8")).hexdigest()[:12]

def build_pair_id(row: pd.Series) -> str:
    ents = row.get("entities") or []
    ents_key = tuple(sorted([(str(ents[0][0]) if ents else "", str(ents[0][1]) if ents else ""),
                             (str(ents[1][0]) if len(ents)>1 else "", str(ents[1][1]) if len(ents)>1 else "")],
                            key=lambda t: (t[0].lower(), t[1].lower())))
    ctx_key = stable_norm_key(str(row.get("context","")))
    parts = [
        str(row.get("question_index","?")),
        str(row.get("question_polarity","?")),
        str(row.get("category","?")),
        str(ents_key),
        ctx_key,
        str(row.get("example_id","?")),
    ]
    return "_".join(parts)

def _first_wb_span(text: str, needle: str):
    if not isinstance(text, str) or not isinstance(needle, str):
        return None
    s = needle.strip()
    if not s:
        return None
    m = re.search(rf"\b{re.escape(s)}\b", text, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None

def naive_swap_first_mentions(context: str, e1: Tuple[str,str], e2: Tuple[str,str]) -> Optional[str]:
    if not isinstance(context, str):
        return None
    c1 = [t for t in (e1[0], e1[1]) if isinstance(t, str) and t.strip()]
    c2 = [t for t in (e2[0], e2[1]) if isinstance(t, str) and t.strip()]
    if not (c1 and c2):
        return None
    A = next((a for a in c1 if _first_wb_span(context, a)), None)
    if not A:
        return None
    A_span = _first_wb_span(context, A)
    # prefer B after A
    B_span = None; B_txt = None
    for b in c2:
        span = _first_wb_span(context, b)
        if span:
            if span[0] > A_span[1]:
                B_span, B_txt = span, b
                break
            if B_span is None:
                B_span, B_txt = span, b
    if not B_span or A_span == B_span:
        return None
    s1,e1_ = A_span; s2,e2_ = B_span; s = context
    if s1 < s2:
        swapped = s[:s1] + s[s2:e2_] + s[e1_:s2] + s[s1:e1_] + s[e2_:]
    else:
        swapped = s[:s2] + s[s1:e1_] + s[e2_:s1] + s[s2:e2_] + s[e1_:]
    return fix_indef_articles(swapped)



# -----------------------------
# Optional RITS (LLM swapping)
# -----------------------------

class RITSClient:
    """Optional: provide to enable LLM swaps. If you don’t need it, pass None."""
    def __init__(self, api_key: Optional[str] = None, cfg_path: str = "config.json"):
        self.api_key = (api_key or os.getenv("RITS_API_KEY") or "").strip()
        if not self.api_key and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.api_key = (json.load(f).get("RITS_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("RITS API key not provided (env RITS_API_KEY or config.json).")
        import requests
        self._requests = requests
        info = requests.get("https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo",
                            headers={"RITS_API_KEY": self.api_key})
        if info.status_code != 200:
            raise RuntimeError(f"Failed RITS model list: {info.status_code} {info.text}")
        self.models = {m["model_name"]: m["endpoint"] for m in info.json()}
        from openai import OpenAI
        self._OpenAI = OpenAI

    def completions(self, model_name: str, prompts: List[str], max_tokens: int = 256, temperature: float = 0.2):
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not on RITS.")
        base_url = f"{self.models[model_name]}/v1"
        client = self._OpenAI(api_key=self.api_key, base_url=base_url, default_headers={"RITS_API_KEY": self.api_key})
        return client.completions.create(model=model_name, prompt=prompts, temperature=temperature, max_tokens=max_tokens)

def swap_order_prompt(context: str, e1: Tuple[str,str], e2: Tuple[str,str]) -> str:
    return (
        "Rewrite the text so that the FIRST mention of the two specified entities swaps order: "
        f"'{e1[0]}' must FIRST appear AFTER '{e2[0]}'.\n"
        "STRICT OUTPUT RULES:\n"
        "1) Output ONLY the rewritten text. No preface, labels, or notes.\n"
        "2) Preserve wording/punctuation; make the MINIMAL edit needed to flip order.\n"
        "3) Do NOT add/remove facts.\n\n"
        f"Original text:\n{context}\n"
    )

def llm_swap_batch(rits: RITSClient, model_name: str, contexts: List[str],
                   e1s: List[Tuple[str,str]], e2s: List[Tuple[str,str]],
                   batch: int = 64, max_tokens: int = 256, temperature: float = 0.2) -> List[str]:
    prompts = [swap_order_prompt(c, a, b) for c,a,b in zip(contexts, e1s, e2s)]
    outs = []
    for i in range(0, len(prompts), batch):
        resp = rits.completions(model_name, prompts[i:i+batch], max_tokens=max_tokens, temperature=temperature)
        # parse .choices[*].text
        for ch in getattr(resp, "choices", []):
            txt = getattr(ch, "text", None) or getattr(getattr(ch, "message", None), "content", "")
            outs.append((txt or "").strip().split("\n",1)[0])
    # length align
    if len(outs) < len(prompts): outs += [""]*(len(prompts)-len(outs))
    return outs[:len(prompts)]

# -----------------------------
# Embedding
# -----------------------------

def batch_embed(texts: List[str], model_name: str, batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if not texts: return np.zeros((0, 384))
    model = SentenceTransformer(model_name)
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        chunk = texts[i:i+batch_size]
        vecs = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        out.append(vecs)
    return np.vstack(out)

# -----------------------------
# Core pipeline
# -----------------------------

def run_rq4_pipeline(df: pd.DataFrame,
                     rits_client: RITSClient | None = None,
                     rits_model_name: str = "ibm-granite/granite-13b-instruct-v2",
                     sample_n: Optional[int] = 1200,
                     only_negative: bool = True,
                     context_condition: Optional[str] = None,   # None | ambig | disambig
                     embed_model: str = "ibm-granite/granite-embedding-30m-english",
                     embed_batch: int = 128,
                     min_pair_checks: bool = True) -> Dict[str, pd.DataFrame]:
    """Return dict of results DataFrames."""

    req = {"example_id","question_index","question_polarity","category","context_condition",
           "context","question","answer_info","additional_metadata"}
    missing = req - set(df.columns)
    if missing: raise ValueError(f"Input missing columns: {sorted(missing)}")

    work = df.copy()

    # ---- filters ----
    if only_negative:
        work = work[work["question_polarity"].astype(str).str.lower().eq("neg")].copy()
    if context_condition in {"ambig","disambig"}:
        work = work[work["context_condition"].astype(str).str.lower().eq(context_condition)].copy()
    if sample_n is not None and len(work) > sample_n:
        work = work.sample(sample_n, random_state=7).reset_index(drop=True)


    # ---- entities + initial order + pair id ----
    work["entities"] = [extract_two_entities_from_row(r) for _, r in work.iterrows()]

    def safe_order(r):
        if isinstance(r["entities"], list) and len(r["entities"]) == 2:
            return entity_order(r["context"], r["entities"][0], r["entities"][1], r["category"])
        return None

    work["orig_order"] = [safe_order(r) for _, r in work.iterrows()]
    work["pair_id"]    = [build_pair_id(r) for _, r in work.iterrows()]


    # ---- debug AFTER columns exist ----
    print("[DBG] work size:", len(work))
    print("[DBG] entities extracted:", work["entities"].apply(lambda x: isinstance(x, list) and len(x)==2).sum())
    print("[DBG] orders detected (A-B/B-A/None):",
        int((work["orig_order"]=="A-B").sum()),
        int((work["orig_order"]=="B-A").sum()),
        int(work["orig_order"].isna().sum()))

    # Peek at schemas (best-effort, won’t crash if column missing)
    try:
        sample_ai = work["answer_info"].head(3).tolist()
        parsed = []
        for x in sample_ai:
            p = _coerce_answer_info(x)
            parsed.append(p if isinstance(p, dict) else ("<parse_failed>", str(x)[:200]))
        print("[DBG] sample answer_info (first 3, coerced):", parsed)
    except Exception:
        pass



    # build pairs per (pair_id, context_condition)
    pairs = []
    to_llm_ctx, to_llm_e1, to_llm_e2, meta_rows = [], [], [], []

    for (pid, ccond), g in work.groupby(["pair_id","context_condition"], dropna=False):
        g = g.reset_index(drop=True)
        row0 = g.iloc[0]
        ents = row0.get("entities")
        if not (isinstance(ents, list) and len(ents) == 2): 
            continue

        orders = set([o for o in g["orig_order"] if o in {"A-B","B-A"}])
        if orders == {"A-B","B-A"}:
            ab = g[g["orig_order"]=="A-B"].iloc[0]
            ba = g[g["orig_order"]=="B-A"].iloc[0]
            pairs.append(("dataset", ab, ba))
            continue

        # need to synthesize opposite order
        row = g.iloc[0]
        e1, e2 = row["entities"]
        # choose which appears first in text (fallback e1,e2)
        ord_now = row.get("orig_order")
        if ord_now == "A-B": eA, eB = e1, e2
        elif ord_now == "B-A": eA, eB = e2, e1
        else: eA, eB = e1, e2

        # programmatic (first)
        prog = programmatic_swap(row["context"], eA, eB, row["category"])
        if prog and verify_flipped(row["context"], prog, eA, eB, row["category"]):
            partner = row.copy()
            partner["context"] = prog
            partner["orig_order"] = "B-A" if ord_now == "A-B" else ("A-B" if ord_now == "B-A" else None)
            partner["source"] = "programmatic"
            pairs.append(("programmatic", row, partner))
            continue

        # after programmatic failure:
        naive = naive_swap_first_mentions(row["context"], eA, eB)
        if naive and verify_flipped(row["context"], naive, eA, eB, row["category"]):
            partner = row.copy()
            partner["context"] = naive
            partner["orig_order"] = "B-A" if ord_now == "A-B" else ("A-B" if ord_now == "B-A" else None)
            partner["source"] = "naive"
            pairs.append(("naive", row, partner))
            continue

        # queue LLM if available
        if rits_client is not None:
            to_llm_ctx.append(row["context"]); to_llm_e1.append(eA); to_llm_e2.append(eB); meta_rows.append(row)

    print("[DBG] pair sources:", Counter([src for src, _, __ in pairs]))
    print("[DBG] total pairs:", len(pairs))

    # do LLM swaps if queued
    if to_llm_ctx and rits_client is not None:
        llm_out = llm_swap_batch(rits_client, rits_model_name, to_llm_ctx, to_llm_e1, to_llm_e2)
    else:
        llm_out = []

    for row, cand in zip(meta_rows, llm_out):
        if not cand: continue
        e1, e2 = row["entities"]
        ord_now = row.get("orig_order")
        if ord_now == "A-B": eA, eB = e1, e2
        elif ord_now == "B-A": eA, eB = e2, e1
        else: eA, eB = e1, e2
        if verify_flipped(row["context"], cand, eA, eB, row["category"]):
            partner = row.copy()
            partner["context"] = cand
            partner["orig_order"] = "B-A" if ord_now == "A-B" else ("A-B" if ord_now == "B-A" else None)
            partner["source"] = "llm"
            pairs.append(("llm", row, partner))

    # materialize final paired table (recompute orders on stored contexts)
    recs = []
    for source, r1, r2 in pairs:
        e1, e2 = r1["entities"]
        o1 = entity_order(r1["context"], e1, e2, r1["category"])
        o2 = entity_order(r2["context"], e1, e2, r2["category"])

        # Coerce stereotyped_groups -> list[str] (handles np.ndarray, tuple, scalar, None)
        sg_raw = (r1.get("additional_metadata") or {}).get("stereotyped_groups", [])
        sg = _to_list(sg_raw)

        recs.append(dict(
            pair_id=r1["pair_id"],
            source=source,
            context_condition=r1["context_condition"],
            category=r1["category"],
            example_id=r1["example_id"],
            question_index=r1["question_index"],
            question=r1["question"],
            context_1=r1["context"], order_1=o1,
            context_2=r2["context"], order_2=o2,
            entities=r1["entities"],
            stereotyped_groups=sg,
        ))

    out = pd.DataFrame(recs)

    if "stereotyped_groups" in out.columns:
        out["stereotyped_groups"] = out["stereotyped_groups"].apply(_to_list)


    # keep only opposite valid orders (loose; no template-equality audit)
    if len(out):
        mask_ok = out["order_1"].isin({"A-B","B-A"}) & out["order_2"].isin({"A-B","B-A"}) & (out["order_1"] != out["order_2"])
        out = out[mask_ok].reset_index(drop=True)

    # ---------- embeddings & Δcos ----------
    if len(out) == 0:
        warnings.warn("No A-B/B-A pairs produced. Try loosening filters or removing --only-negative.")
        return {"pairs": out}

    q_list = sorted(out["question"].dropna().unique().tolist())
    q_vecs = batch_embed(q_list, embed_model, batch_size=embed_batch)
    q_map = {q: q_vecs[i] for i, q in enumerate(q_list)}

    c_texts = []
    for _, r in out.iterrows():
        c_texts.extend([r["context_1"], r["context_2"]])
    c_vecs = batch_embed(c_texts, embed_model, batch_size=embed_batch)

    sims1, sims2 = [], []
    for i, r in out.iterrows():
        qv = q_map[r["question"]]
        sims1.append(float(np.dot(qv, c_vecs[2*i + 0])))
        sims2.append(float(np.dot(qv, c_vecs[2*i + 1])))
    out["cos_sim_1"] = sims1
    out["cos_sim_2"] = sims2
    out["delta_sim"] = np.where(
        out["order_1"].eq("A-B"), out["cos_sim_1"] - out["cos_sim_2"],
        out["cos_sim_2"] - out["cos_sim_1"]
    )

    # ---------- positional flags ----------
    def first_is_stereotyped(row) -> bool:
        # groups can be list/tuple/set/np.ndarray/scalar; normalize to list of strings
        groups_list = _to_list(row.get("stereotyped_groups"))
        groups = {str(g).strip().lower() for g in groups_list if str(g).strip()}
        if not groups:
            return False

        # pick the FIRST entity according to order_1
        e1, e2 = row["entities"]
        first = e1 if row["order_1"] == "A-B" else e2
        name = str(first[0]).strip().lower()
        label = str(first[1]).strip().lower()
        cat = str(row.get("category", "")).strip().lower()

        # small gender expansion
        if "gender" in cat:
            expanded = set()
            for g in groups:
                if g == "f":
                    expanded |= {"woman", "girl"}
                elif g == "m":
                    expanded |= {"man", "boy"}
                elif g in {"trans", "transgender"}:
                    expanded |= {"trans", "transgender"}
                else:
                    expanded.add(g)
            groups = expanded

        text = f"{name} {label}"
        return any(g and g in text for g in groups)


    out["first_is_stereotyped"] = out.apply(first_is_stereotyped, axis=1)

    def stat_block(vals: pd.Series) -> dict:
        x = pd.to_numeric(vals, errors="coerce").dropna().values
        if len(x) == 0:
            return {"N":0,"mean":np.nan,"std":np.nan,"t":np.nan,"p_t":np.nan,"w":np.nan,"p_w":np.nan}
        t, p = ttest_1samp(x, 0.0)
        try:
            w, pw = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
        except Exception:
            w, pw = np.nan, np.nan
        return {"N":int(len(x)),"mean":float(np.mean(x)),"std":float(np.std(x, ddof=1) if len(x)>1 else 0.0),
                "t":float(t),"p_t":float(p),"w":float(w) if not np.isnan(w) else np.nan,"p_w":float(pw) if not np.isnan(pw) else np.nan}

    # summaries
    overall = pd.DataFrame([{
        "N_pairs": len(out),
        "mean_delta": out["delta_sim"].mean(),
        "std_delta": out["delta_sim"].std(ddof=1),
        "t_stat": float(ttest_1samp(out["delta_sim"], 0.0).statistic),
        "p_value": float(ttest_1samp(out["delta_sim"], 0.0).pvalue),
    }])

    by_cat = (
        out.groupby("category", as_index=False)["delta_sim"]
           .agg(N="count", mean_delta="mean", std_delta="std")
           .sort_values("mean_delta", ascending=False)
    )

    pos_overall_rows = []
    for flag, g in out.groupby("first_is_stereotyped"):
        s = stat_block(g["delta_sim"])
        s["first_is_stereotyped"] = bool(flag)
        pos_overall_rows.append(s)
    positional_overall = pd.DataFrame(pos_overall_rows).sort_values("mean", ascending=False)

    pos_cat_rows = []
    for (cat, flag), g in out.groupby(["category","first_is_stereotyped"]):
        s = stat_block(g["delta_sim"])
        s["category"] = cat
        s["first_is_stereotyped"] = bool(flag)
        pos_cat_rows.append(s)
    positional_by_category = pd.DataFrame(pos_cat_rows).sort_values(["category","mean"], ascending=[True, False])

    # quick plot
    try:
        plt.figure(figsize=(12,5))
        plt.bar(by_cat["category"], by_cat["mean_delta"], alpha=0.9)
        plt.axhline(0, color="black", linestyle="--")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean Δcos (A-B − B-A) to Negative Question")
        plt.title("RQ4: Entity Order Effect by Category")
        plt.tight_layout()
    except Exception:
        warnings.warn("Plotting failed; continuing without figure.")

    return {
        "pairs": out,
        "overall": overall,
        "by_category": by_cat,
        "positional_overall": positional_overall,
        "positional_by_category": positional_by_category,
    }

# -----------------------------
# CLI
# -----------------------------

def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".csv"):     return pd.read_csv(path)
    if path.endswith(".json"):    return pd.read_json(path, lines=False)
    raise ValueError("Unsupported input format. Use .parquet / .csv / .json")

def main():
    ap = argparse.ArgumentParser(description="BBQ RQ4 (Entity position effects)")
    ap.add_argument("--input", required=True, help="Path to BBQ data (parquet/csv/json)")
    ap.add_argument("--output-dir", required=True, help="Directory to write results")
    ap.add_argument("--sample-n", type=int, default=500000, help="Sample size (set 0 to use all)")
    ap.add_argument("--only-negative", action="store_true", help="Use only negative polarity questions")
    ap.add_argument("--context-condition", choices=["ambig","disambig"], default=None,
                    help="Optional filter to one condition")
    ap.add_argument("--embed-model", default="ibm-granite/granite-embedding-30m-english", help="Sentence-Transformers model name")
    ap.add_argument("--embed-batch", type=int, default=128)
    ap.add_argument("--use-rits", action="store_true", help="Enable RITS-based LLM swapping")
    ap.add_argument("--rits-model", default="ibm-granite/granite-13b-instruct-v2")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = read_any(args.input)

    rits = None
    if args.use_rits:
        try:
            rits = RITSClient()
            print("[RITS] enabled.")
        except Exception as e:
            warnings.warn(f"RITS disabled: {e}")
            rits = None

    df = pd.read_parquet("out/granite.parquet")
    row = df.iloc[0]
    from BBQ import _coerce_answer_info, extract_two_entities_from_row  # or however you import in your env
    ai = _coerce_answer_info(row["answer_info"])
    print(type(ai), list(ai.keys())[:5] if isinstance(ai, dict) else ai)
    print(extract_two_entities_from_row(row))

    res = run_rq4_pipeline(
        df=df,
        rits_client=rits,
        rits_model_name=args.rits_model,
        sample_n=None if args.sample_n == 0 else args.sample_n,
        only_negative=args.only_negative,
        context_condition=args.context_condition,
        embed_model=args.embed_model,
        embed_batch=args.embed_batch,
    )


    # write outputs
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            outp = os.path.join(args.output_dir, f"rq4_{k}.csv")
            v.to_csv(outp, index=False)
            print(f"[write] {k}: {outp}")

    # save plot if created
    try:
        figpath = os.path.join(args.output_dir, "rq4_by_category.png")
        plt.savefig(figpath, dpi=180, bbox_inches="tight")
        print(f"[write] plot: {figpath}")
    except Exception:
        pass

if __name__ == "__main__":
    main()