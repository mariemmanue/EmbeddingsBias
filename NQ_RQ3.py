from datasets import load_dataset, concatenate_datasets
import os, math, re, json, argparse
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless environments
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util as st_util
from scipy import stats
import matplotlib.pyplot as plt
import warnings

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


import requests, difflib
from dataclasses import dataclass

import torch
print("[DEBUG] CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[DEBUG] Using GPU:", torch.cuda.get_device_name(0))

try:
    with open("config.json") as _f:
        _jcfg = json.load(_f)
except Exception:
    _jcfg = {}

# --- RITS embedding backend (add to both files) ---

try:
    from openai import OpenAI as _OpenAI  # OpenAI-compatible client for RITS
except Exception:
    _OpenAI = None

class RITSClient:
    """Minimal client to discover endpoints and create an OpenAI-compatible client."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("RITS API key missing")
        self.api_key = api_key
        self._models = None
        self._openai = None

    @property
    def models(self) -> dict:
        if self._models is None:
            url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
            resp = requests.get(url, headers={"RITS_API_KEY": self.api_key}, timeout=30)
            resp.raise_for_status()
            self._models = {m["model_name"]: m["endpoint"] for m in resp.json()}
        return self._models

    def _get_openai(self):
        if self._openai is None:
            if _OpenAI is None:
                raise RuntimeError("openai>=1.0 is required for RITS. pip install openai")
            self._openai = _OpenAI
        return self._openai

    def client_for(self, model_name: str):
        # Validates the model exists in self.models, builds an OpenAI-compatible client pointed at <endpoint>/v1, passes API key in both the param and default header.
        if model_name not in self.models:
            avail = sorted(self.models.keys())[:12]
            raise KeyError(f"Unknown RITS model: '{model_name}'. Available (sample): {avail} ...")
        base_url = f"{self.models[model_name]}/v1"
        OpenAI = self._get_openai()
        return OpenAI(api_key=self.api_key, base_url=base_url, default_headers={"RITS_API_KEY": self.api_key})

    def batched_generate(
        self,
        model_name: str,
        prompts: List[str],
        max_tokens: int = 120,
        temperature: float = 0.2,
        batch_size: int = 64,
        progress: bool = True,
    ) -> List[str]:
        """
        OpenAI-compatible batched text generation on RITS.
        Tries chat.completions first; falls back to completions if chat isn't supported.
        Returns one string per prompt, preserving order.
        """
        client = self.client_for(model_name)
        outs: List[str] = []

        it = range(0, len(prompts), batch_size)
        if progress:
            from tqdm.auto import tqdm as _tqdm
            it = _tqdm(it, total=(len(prompts) + batch_size - 1) // batch_size, desc="RITS gen batches")

        for i in it:
            chunk = prompts[i:i + batch_size]
            try:
                # Preferred path for Granite instruct models
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": p} for p in chunk],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                outs.extend([(c.message.content or "") for c in resp.choices])
            except Exception:
                # Fallback for non-chat endpoints
                resp = client.completions.create(
                    model=model_name,
                    prompt=chunk,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                outs.extend([(c.text or "") for c in resp.choices])

        return outs


class RITSEmbedder:
    """
   replacement for SentenceTransformer with an .encode() method.
    Uses RITS /embeddings endpoint. Returns np.ndarray shape (N, D).
    """
    # Creates a RITS client for the embedding model.
    def __init__(self, rits: RITSClient, model_name: str, device: str = "cuda"):
        self.rits = rits
        self.model_name = model_name
        self.client = rits.client_for(model_name)
        self.device = device  # for parity with ST; not used by RITS

    def encode(self, texts: List[str], batch_size: int = 64, convert_to_numpy: bool = True,
               show_progress_bar: bool = False, **_):
        # RITS is OpenAI-compatible: client.embeddings.create(model=..., input=[...])
        # We manually batch to respect service limits.
        outs = []
        rng = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm.auto import tqdm
            rng = tqdm(rng, total=(len(texts)+batch_size-1)//batch_size, desc="Batches")
        for i in rng:
            chunk = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=chunk)
            # resp.data is list with .embedding vectors in input order
            outs.extend([np.asarray(e.embedding, dtype=np.float32) for e in resp.data])
        arr = np.vstack(outs) if outs else np.zeros((0, 0), dtype=np.float32)
        return arr if convert_to_numpy else arr.tolist()

def l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        denom = np.linalg.norm(arr) + 1e-12
        return arr / denom
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / denom

def batch_embed_any(texts: List[str], model, batch_size: int = 64,
                    normalize: bool = True, progress: bool = False, desc: str = "Embedding") -> np.ndarray:
    """
    Works for both SentenceTransformer and RITSEmbedder (they both expose .encode()).
    """
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        show_progress_bar=progress)
    return l2_normalize(vecs) if normalize else vecs

def load_embedder_from_args(args):
    """
    If --use-rits-embed is set, return RITSEmbedder(args.rits_embed_model, args.rits_key).
    Otherwise, load SentenceTransformer(args.model) as before.
    """
    if getattr(args, "use_rits_embed", False):
        print(f"[MODEL] Using RITS embeddings: {args.rits_embed_model}")
        rits = RITSClient(api_key=args.rits_key)
        _ = rits.models
        return RITSEmbedder(rits, args.rits_embed_model, device="cuda")
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MODEL] Loading SentenceTransformer on {dev}: {args.model}")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(args.model, device=dev)


# ------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

UNKNOWN_ANSWERS = {
    "can't be determined", "not enough info", "unknown",
    "can't answer", "cannot answer", "cannot be determined",
    "not known", "undetermined"
}

# ====================================================
# Load Dataset
# ====================================================
TARGET_SUBSETS = [
    "Age","Disability_status","Gender_identity","Nationality","Physical_appearance",
    "Race_ethnicity","Race_x_SES","Race_x_gender","Religion","SES","Sexual_orientation"
]

ELFSONG_SPLITS = {
    "age":"Age",
    "disability_status":"Disability_status",
    "gender_identity":"Gender_identity",
    "nationality":"Nationality",
    "physical_appearance":"Physical_appearance",
    "race_ethnicity":"Race_ethnicity",
    "race_x_ses":"Race_x_SES",
    "race_x_gender":"Race_x_gender",
    "religion":"Religion",
    "ses":"SES",
    "sexual_orientation":"Sexual_orientation"
}
WALLEDAI_SPLITS = {
    "age":"Age",
    "disabilityStatus":"Disability_status",
    "genderIdentity":"Gender_identity",
    "nationality":"Nationality",
    "physicalAppearance":"Physical_appearance",
    "raceEthnicity":"Race_ethnicity",
    "raceXSes":"Race_x_SES",
    "raceXGender":"Race_x_gender",
    "religion":"Religion",
    "ses":"SES",
    "sexualOrientation":"Sexual_orientation"
}

def load_bbq_all() -> pd.DataFrame:
    """Load BBQ from mirrors that DO NOT require dataset scripts."""
    tried = []
    for repo, splits in [("Elfsong/BBQ", ELFSONG_SPLITS), ("walledai/BBQ", WALLEDAI_SPLITS)]:
        try:
            print(f"[BBQ] Trying repo: {repo}")
            parts = []
            for split_name, category in splits.items():
                ds = load_dataset(repo, split=split_name)
                if "category" in ds.column_names:
                    ds = ds.remove_columns(["category"]).add_column("category", [category] * ds.num_rows)
                else:
                    ds = ds.add_column("category", [category] * ds.num_rows)
                parts.append(ds)
                print(f"  loaded split={split_name:<20} → {category:<22} rows={ds.num_rows}")
            bbq_all = concatenate_datasets(parts)
            pdf = bbq_all.to_pandas()
            print(f"[BBQ] Total rows: {len(pdf)} | columns: {list(pdf.columns)[:20]}")
            return pdf
        except Exception as e:
            tried.append((repo, str(e)))
            print(f"[BBQ] Failed: {repo} → {e}")
    raise RuntimeError(f"Could not load any BBQ mirror; tried={tried}")

def load_nq_any():
    tried = []
    specs = [
        ("sentence-transformers/natural-questions", {"split": "train"}),
        ("natural_questions_open", {"split": "train"}),
        ("Tevatron/nq", {"name": "train", "split": "train"})
    ]
    for repo, kwargs in specs:
        try:
            print(f"[NQ] Trying {repo} {kwargs} ...")
            ds = load_dataset(repo, **kwargs)
            print(f"[NQ] Loaded: {repo}, rows={len(ds)}")
            return ds
        except Exception as e:
            tried.append((repo, str(e)))
            print(f"[NQ] Failed: {repo} → {e}")
    raise RuntimeError(f"Could not load any NQ mirror. Tried: {tried}")

def load_granite(out_dir: str) -> pd.DataFrame:
    # Reads granite.parquet else granite.csv.
    csv = os.path.join(out_dir, "granite.csv")
    pq  = os.path.join(out_dir, "granite.parquet")

    _arr_pat = re.compile(r"array\(\s*(\[[^\]]*\])\s*,\s*dtype=object\s*\)")
    def _strip_numpy_arrays(s: str) -> str:
        return _arr_pat.sub(r"\1", s)

    def _string_to_jsonable(s: str) -> str:
        s = (s or "").strip()
        if not s or (s[0] not in "{[" or s[-1] not in "}]"):
            return ""
        s = s.replace('""', '"')
        s = _strip_numpy_arrays(s)
        s = s.replace("'", '"')
        return s

    def _parse_obj_cell(x):
        if isinstance(x, dict):
            return x
        if not isinstance(x, str):
            return {}
        js = _string_to_jsonable(x)
        if not js:
            return {}
        try:
            return json.loads(js)
        except Exception:
            return {}

    def _coerce_cols(df):
        if "answer_info" in df.columns:
            df["answer_info"] = df["answer_info"].apply(_parse_obj_cell)
        if "additional_metadata" in df.columns:
            df["additional_metadata"] = df["additional_metadata"].apply(_parse_obj_cell)
        return df

    if os.path.exists(pq):
        print(f"[LOAD] Loading parquet → {pq}")
        df = pd.read_parquet(pq)
        return _coerce_cols(df)

    if os.path.exists(csv):
        print(f"[LOAD] Loading csv → {csv}")
        df = pd.read_csv(csv, dtype={"question_index":"Int64"}, na_filter=False)
        return _coerce_cols(df)

    raise FileNotFoundError("No granite.{parquet,csv} found")

def safe_json(s):
    if isinstance(s, str):
        s = s.strip()
        if s and (s[0] in "[{"):
            try:
                return json.loads(s)
            except Exception:
                return s
    return s

# ====================================================
# Cosine similarity over BBQ 
# ====================================================
def get_cosine_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    embs = model.encode([text1 or "", text2 or ""], convert_to_tensor=True)
    return st_util.pytorch_cos_sim(embs[0], embs[1]).item()

def cos_sim_single(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# ====================================================
# plots
# ====================================================

def plot_rq3_generation_cosine(agg_df: pd.DataFrame, out_path: str):
    """
    Bar plot of cosine(base, injected) by identity_category with 95% CI error bars
    for begin vs end injection positions. Expects columns:
      identity_category, mean_begin, mean_end,
      ci_begin_lo, ci_begin_hi, ci_end_lo, ci_end_hi
    """
    if agg_df is None or agg_df.empty:
        print(f"[PLOT:RQ3:GEN_COS] nothing to plot for {out_path}")
        return

    x = np.arange(len(agg_df))
    w = 0.38

    # error bar half-widths from CI bounds
    eb_begin = [agg_df["mean_begin"] - agg_df["ci_begin_lo"],
                agg_df["ci_begin_hi"] - agg_df["mean_begin"]]
    eb_end   = [agg_df["mean_end"] - agg_df["ci_end_lo"],
                agg_df["ci_end_hi"] - agg_df["mean_end"]]

    plt.figure(figsize=(14, 6))
    plt.bar(x - w/2, agg_df["mean_begin"], width=w, label="begin",
            yerr=eb_begin, capsize=3, alpha=0.9)
    plt.bar(x + w/2, agg_df["mean_end"],   width=w, label="end",
            yerr=eb_end, capsize=3, alpha=0.9)

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, agg_df["identity_category"], rotation=45, ha="right")
    plt.ylabel("Cosine(base, injected)")
    plt.title("RQ3 Generation Drift (semantic): Cosine similarity to base (±95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[PLOT:RQ3:GEN_COS] wrote → {out_path}")

# ====================================================
# Text/identity helpers (injection)
# ====================================================
VOWEL_SOUND_EXCEPTIONS_AN = ("honest","hour","honor","heir","heirloom")
CONSONANT_SOUND_EXCEPTIONS_A = ("university","unilateral","euro","user","ute")

def needs_an(word: str) -> bool:
    w = (word or "").strip().lower()
    if not w: return False
    if any(w.startswith(x) for x in VOWEL_SOUND_EXCEPTIONS_AN): return True
    if any(w.startswith(x) for x in CONSONANT_SOUND_EXCEPTIONS_A): return False
    return w[0] in "aeiou"

def choose_article(phrase: str) -> str:
    first = (phrase or "").strip().split()[0] if phrase else ""
    return "an" if needs_an(first) else "a"

def ensure_terminal(s: str) -> str:
    s = (s or "").rstrip()
    return s if s.endswith((".", "!", "?")) else s + "."

def looks_like_name(entity: str) -> bool:
    if not entity or not isinstance(entity, str): return False
    toks = entity.strip().split()
    if len(toks) < 2: return False
    role_heads = {"man","woman","boy","girl","person","people"}
    if any(t.lower() in role_heads for t in toks): return False
    uppers = sum(t[:1].isupper() for t in toks if t)
    return uppers >= max(2, int(0.7 * len(toks)))

def is_full_np(s: str) -> bool:
    if not s: return False
    e = s.strip().lower()
    return e.startswith(("a ","an ","the ","this ","that ","these ","those ",
                         "person ","person-","man","woman","boy","girl","teen",
                         "elderly","older","younger","middle-aged","pregnant",
                         "disabled","deaf","blind","wheelchair"))

def normalize_ses(lab: str) -> str:
    SES_MAP = {"highses":"wealthy","lowses":"low-income","rich":"wealthy","poor":"low-income"}
    base = re.sub(r"[_\-]+","",(lab or "").lower())
    return SES_MAP.get(base, lab)

JOB_SUFFIXES = ("er","or","ist","ian","ess")
ROLE_HEADS = {"man","woman","boy","girl","person","neighbor","acquaintance","guy"}
THING_HINTS = {"backpack","record","collection","jewelry","tattoo","piercing","scar","scars"}
APPEARANCE_HINTS = {"eyes","hair","afro","dreadlocks","tattoo","piercing","piercings","scar","scars","jewelry"}
RELIG_DEMONYM_CATS = {"religion","nationality","race_ethnicity","race_x_gender","race_x_ses"}
TITLECASE_STOP = {"Christian","Catholic","Protestant","Muslim","Sikh","Hindu",
                  "French","German","Mexican","Italian","Australian","Vietnamese","Libyan",
                  "Bangladeshi","Swedish","Slovak"}

def is_single_title_name(ent: str, category: str) -> bool:
    if not ent or " " in ent: return False
    if not ent[:1].isupper(): return False
    if ent in TITLECASE_STOP: return False
    if category.strip().lower() in RELIG_DEMONYM_CATS: return False
    return ent.lower() not in ROLE_HEADS

def normalize_appearance_piece(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\b([a-z]+)-haired\b", r"\1 hair", s)
    return s

def parse_gender_head(entity: str, label: str) -> str:
    gender = None
    if isinstance(label, str) and "-" in label:
        g = label.split("-",1)[0].strip().lower()
        if g in {"f","female","woman","girl"}: gender = "woman"
        elif g in {"m","male","man","boy"}: gender = "man"
    e = (entity or "").lower()
    if "boy" in e: return "boy"
    if "girl" in e: return "girl"
    return gender or "person"

def realize_identity_ref(entity: str, label: str, category: str) -> Tuple[str, bool]:
    ent = (entity or "").strip()
    lab = (label or "").strip()
    cat = (category or "").strip().lower()

    if looks_like_name(ent) or is_single_title_name(ent, cat):
        return ent, True

    low = ent.lower()
    if low.startswith(("who ","with ","using ","was ","is ","are ","has ","have ")):
        ent_clean = low
        if not ent_clean.startswith("who "):
            ent_clean = "who " + ent_clean
        return "a person " + ent_clean, False

    if re.fullmatch(r"[a-z]+ly", low):
        return "a person who dresses " + low, False
    if low in {"skinny","tall","short","fat","chubby","slim"}:
        return f"a {low} person", False

    if any(low.endswith(suf) for suf in JOB_SUFFIXES) and low not in {"low","high"}:
        return f"{choose_article(ent)} {ent}", False

    if cat == "age":
        m = re.search(r"(\d+)[ -]?year[- ]?old", low)
        if m:
            n = m.group(1)
            return f"{('an' if needs_an(n) else 'a')} {n}-year-old", False
        if ent and (is_full_np(ent) or any(h in low for h in ROLE_HEADS)):
            return (ent if ent.lower().startswith(("a ","an ","the ")) else f"{choose_article(ent)} {ent}"), False
        if ent:
            return f"{choose_article(ent)} {ent}", False
        return "an older adult" if "old" in lab.lower() else "a younger person", False

    if cat == "disability_status":
        if any(k in low for k in ["wheelchair","wheel chair","chair user"]):
            return "a wheelchair user", False
        if low in {"blind","deaf"}:
            return f"a person who is {low}", False
        if low in {"hard of hearing"}:
            return "a hard of hearing person", False
        if low in {"depressed","bipolar"}:
            return "a person who is " + low, False
        if low in {"autism","autism spectrum disorder","ptsd","bpd","schizophrenia",
                   "anxiety","depression","cerebral palsy"}:
            return "a person with " + low, False
        if any(h in low for h in THING_HINTS):
            return "a person with " + ent.lower(), False
        if is_full_np(ent):
            return (ent if ent.lower().startswith(("a ","an ","the ")) else f"{choose_article(ent)} {ent}"), False
        if ent:
            return f"{choose_article(ent)} {ent} person", False
        return "a disabled person", False

    if cat == "gender_identity":
        if lab.startswith("trans"):
            return ("a transgender woman" if lab.endswith("_F") else "a transgender man"), False
        if low in {"man","woman","boy","girl"}:
            return f"{choose_article(ent)} {ent}", False
        if ent and is_full_np(ent):
            return (ent if ent.lower().startswith(("a ","an ","the ")) else f"{choose_article(ent)} {ent}"), False
        if ent:
            return f"{choose_article(ent)} {ent}", False
        return "a person", False

    if cat == "religion":
        if ent:
            return (ent if ent.lower().startswith(("a ","an ","the ")) else f"{choose_article(ent)} {ent} person"), False
        if lab:
            return f"{choose_article(lab)} {lab} person", False
        return "a religious person", False

    if cat == "ses":
        base = normalize_ses(lab) if lab else ""
        if ent:
            lowent = ent.lower()
            if any(h in lowent for h in ["family","worker","driver","cashier","clerk","engineer",
                                         "scientist","janitor","server","bartender","receptionist","attendant"]):
                return (ent if ent.lower().startswith(("a ","an ","the ")) else f"{choose_article(ent)} {ent}"), False
            return f"{choose_article(ent)} {ent}", False
        if base:
            if base.lower() in {"wealthy","highses","rich"}:
                return "a wealthy person", False
            if base.lower() in {"low-income","lowses","poor","working-class"}:
                return ("a working-class person" if "working" in base.lower() else "a low-income person"), False
        return "a person", False

    if cat == "physical_appearance":
        low = normalize_appearance_piece(low)
        if any(kw in low for kw in APPEARANCE_HINTS):
            return "a person with " + low, False
        if "pregnant" in low:
            return "a pregnant person", False
        if low in {"tall","short"}:
            return f"a {low} person", False
        if ent:
            return (ent if is_full_np(ent) and ent.lower().startswith(("a ","an ","the "))
                    else f"{choose_article(ent)} {ent}"), False
        return "a person", False

    if cat in {"race_ethnicity","nationality"}:
        if ent:
            if any(h in low for h in {"man","woman","boy","girl"}):
                return (ent if ent.lower().startswith(("a ","an ","the ")) else f"{choose_article(ent)} {ent}"), False
            return f"{choose_article(ent)} {ent} person", False
        if lab:
            return f"{choose_article(lab)} {lab} person", False
        return "a person", False

    if ent:
        if any(h in low for h in ROLE_HEADS):
            return (ent if ent.lower().startswith(("a ","an ","the ")) else f"{choose_article(ent)} {ent}"), False
        return f"{choose_article(ent)} {ent} person", False
    return "a person", False

def inject_begin(answer: str, entity: str, label: str, category: str) -> str:
    ans = (answer or "").strip()
    surf, _ = realize_identity_ref(entity, label, category)
    prefix = f"According to {surf}, "
    if ans:
        ans = ans[0].upper() + ans[1:]
    return prefix + ans

def inject_end(answer: str, entity: str, label: str, category: str) -> str:
    surf, _ = realize_identity_ref(entity, label, category)
    return ensure_terminal(answer) + f" According to {surf}."

# ====================================================
# Ranking helpers
# ====================================================
def _ndcg_from_rank(rank: int) -> float:
    # rank is 1-based
    return 1.0 / math.log2(rank + 1)


def _argsort_desc(scores: List[float]) -> List[int]:
    # Stable mergesort indices for descending scores.
    # Returns indices that would sort scores descending.	
    # Uses NumPy’s stable “mergesort” to preserve original order for ties. The trick -s flips the sign so a normal ascending sort becomes descending.
    arr = np.asarray(scores, float)
    return list(np.argsort(-arr, kind="mergesort"))

def _rank_of_positive(sim_pos: float, sim_negs: List[float]) -> int:
    #Rank = 1 + number of negatives ≥ positive (ties counted pessimistically).
    # Given one positive similarity score and a list of negative scores, returns the rank (1‑based) at which the positive would appear if everything were sorted descending.	
    # Counts how many negatives are greater or equal to sim_pos (ties count as ahead). 
    # Adds 1 for the positive itself → rank.
    greater = sum(1 for s in sim_negs if s >= sim_pos)  # ties count ahead (pessimistic)
    return int(greater + 1)

@dataclass # --- define data model, NQ pairs JSONL (query, 1 positive doc, many negatives)
class NQItem:
    """A single training/evaluation instance for neural‑query ranking.

    Attributes
    ----------
    qid : str
        Unique identifier for the query (useful when you want to keep track of
        which result belongs to which query).
    query : str
        The raw text of the query.
    pos_doc : str
        The ground‑truth document that should be ranked highest.  In a binary
        relevance setting this is treated as relevance score = 1.
    neg_docs : List[str]
        A list of “negative” documents – i.e. distractors that are *not*
        relevant to the query.  Each element will be scored against the
        query and used for ranking metrics such as NDCG, MAP, etc.
    """
    qid: str
    query: str
    pos_doc: str
    neg_docs: List[str]

# ====================================================
# Identity harvesting from granite
# ====================================================
def harvest_bbq_pairs_from_granite(
    granite_df: pd.DataFrame,
    max_pairs: Optional[int] = None,
    per_category: Optional[int] = 10,
    shuffle_within_cat: bool = True,
    seed: int = 42,
    progress: bool = False,
) -> List[Tuple[str, str, str]]:
    rng = np.random.default_rng(seed)
    def norm(s): return re.sub(r"\s+"," ",str(s or "").strip())

    pairs_by_cat: Dict[str, List[Tuple[str, str, str]]] = {}
    it = granite_df.itertuples(index=False)
    it = tqdm(it, total=len(granite_df), desc="Harvesting identities", disable=not progress)

    raw_count = 0
    for row in it:
        row = row._asdict()
        cat = row.get("category", "unknown")
# Scans each row’s answer_info (dict of ans* → [entity, label]) and additional_metadata['stereotyped_groups'] to gather pairs.
        ai = row.get("answer_info", {}) or {}
        if isinstance(ai, dict):
            for _, v in ai.items():
                if isinstance(v, np.ndarray): v = v.tolist()
                if isinstance(v, (list, tuple)):
                    if len(v) == 2:
                        ent, lab = v[0], v[1]
                    elif len(v) >= 3:
                        ent, lab = v[-2], v[-1]
                    else:
                        continue
                    ent = norm(ent); lab = norm(lab)
                    if not ent or not lab: 
                        continue
                    if ent.lower() in UNKNOWN_ANSWERS or lab.lower() in UNKNOWN_ANSWERS:
                        continue
                    pairs_by_cat.setdefault(cat, []).append((ent, lab, cat))
                    raw_count += 1

        am = row.get("additional_metadata", {}) or {}
        groups = am.get("stereotyped_groups", [])
        if isinstance(groups, np.ndarray):
            groups = groups.tolist()
        if isinstance(groups, (list, tuple)):
            for g in groups:
                lab = norm(g)
                if lab and lab.lower() not in UNKNOWN_ANSWERS:
                    pairs_by_cat.setdefault(cat, []).append(("", lab, cat))
                    raw_count += 1

    for cat, lst in list(pairs_by_cat.items()):
        seen = set(); uniq = []
        for p in lst:
            if p not in seen:
                seen.add(p); uniq.append(p)
        if shuffle_within_cat:
            rng.shuffle(uniq)
        pairs_by_cat[cat] = uniq # Dedups per category, optional shuffle, returns either first per_category per cat or the full list.

    total_uniq = sum(len(v) for v in pairs_by_cat.values()) # Dedups per category, optional shuffle, returns either first per_category per cat or the full list.
    print(f"[RQ3] harvested raw pairs: {raw_count} across {len(pairs_by_cat)} categories")
    print(f"[RQ3] harvested uniq pairs: {total_uniq} across {len(pairs_by_cat)} categories")

    if per_category is not None: # if category limit
        out = []
        for cat in pairs_by_cat:
            out.extend(pairs_by_cat[cat][:per_category])
        return out

    all_pairs = []
    for cat in pairs_by_cat:
        all_pairs.extend(pairs_by_cat[cat])
    if max_pairs is not None:
        return all_pairs[:max_pairs]
    return all_pairs

def load_nq_pairs_jsonl(path: str, max_items: Optional[int] = None) -> List[NQItem]:
    """Reads line-delimited JSON; tolerates some malformed lines by trimming trailing commas and retrying.

Expects:
- query (string),
- positives.docs (first element used as positive),
- negatives.docs (list).
Returns up to max_items NQItems.

Logs number skipped and loaded."""
    items = []
    bad = 0
    def _try_fix(s: str) -> Optional[dict]:
        # VERY conservative “fix”: strip trailing commas and stray \r,
        t = s.strip().rstrip(",\r\n")
        try:
            return json.loads(t)
        except Exception:
            return None

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                j = _try_fix(line)
                if j is None:
                    bad += 1
                    if bad <= 5:
                        print(f"[NQ JSONL] skip malformed line {ln} (showing first 200 chars): {line[:200]!r}")
                    continue

            qid = str(j.get("query_id", j.get("id", len(items))))
            query = str(j.get("query","")).strip()
            pos_doc = ""
            pos = j.get("positives") or j.get("positive") or {}
            pos_docs = (pos.get("docs") or []) if isinstance(pos, dict) else []
            if pos_docs:
                pos_doc = str(pos_docs[0])
            neg_docs = []
            neg = j.get("negatives") or j.get("negative") or {}
            neg_docs = [str(x) for x in (neg.get("docs") or [])] if isinstance(neg, dict) else []

            if query and pos_doc:
                items.append(NQItem(qid=qid, query=query, pos_doc=pos_doc, neg_docs=neg_docs))
                if max_items and len(items) >= max_items:
                    break

    if bad:
        print(f"[NQ JSONL] skipped {bad} malformed line(s) in {path}")
    print(f"[NQ JSONL] loaded {len(items)} items from {path}")
    return items



# --- injection wrappers for queries/passages
def inject_query(q: str, entity: str, label: str, category: str, where: str) -> str:
    return inject_begin(q, entity, label, category) if where == "begin" else (
           inject_end(q, entity, label, category) if where == "end" else q)

def inject_passage(p: str, entity: str, label: str, category: str, where: str) -> str:
    return inject_begin(p, entity, label, category) if where == "begin" else (
           inject_end(p, entity, label, category) if where == "end" else p)

# ====================================================
# Retrieval eval (queries)
# ====================================================
def evaluate_retrieval_queries(
    items: List[NQItem],
    embed_model: SentenceTransformer,
    pairs: List[Tuple[str,str,str]],
    begin_and_end: bool = True,
    topk: int = 3,
    embed_batch_size: int = 512,
    progress: bool = True
) -> pd.DataFrame:
    rows = []
    it = tqdm(items, desc="RQ3: Retrieval (queries)", disable=not progress)

    for ex in it:
        # Embed docs once: [pos] + negs
        docs = [ex.pos_doc] + list(ex.neg_docs)
        dvecs = batch_embed_any(docs, embed_model, batch_size=embed_batch_size, progress=False)
        # Embed base query once
        q_base_vec = batch_embed_any([ex.query], embed_model, batch_size=embed_batch_size, progress=False)[0]
        # Cosine via dot (normalized vectors)
        sims_base = dvecs @ q_base_vec

        # Keep only if base positive is strictly top-1
        if not (len(sims_base) >= 2 and sims_base[0] > np.max(sims_base[1:])):
            continue

        base_rank  = _rank_of_positive(float(sims_base[0]), sims_base[1:].tolist())
        base_ndcg  = _ndcg_from_rank(base_rank)
        base_order = _argsort_desc(sims_base.tolist())

        # Build all injected queries (batch)
        where_list = (["begin","end"] if begin_and_end else ["begin"])
        inj_specs, inj_texts = [], []
        for where in where_list:
            for ent, lab, cat in pairs:
                disp, _ = realize_identity_ref(ent, lab, cat)
                inj_specs.append((where, ent, lab, cat, disp))
                inj_texts.append(inject_query(ex.query, ent, lab, cat, where))

        # Embed all injected queries at once
        q_inj_mat = batch_embed_any(inj_texts, embed_model, batch_size=embed_batch_size, progress=False)

        # Self drift vs base (per injected query)
        drift_all = 1.0 - (q_inj_mat @ q_base_vec)

        # Similarities to all docs
        sims_all = q_inj_mat @ dvecs.T

        for row_i, (where, ent, lab, cat, disp) in enumerate(inj_specs):
            sims = sims_all[row_i, :]
            inj_rank = _rank_of_positive(float(sims[0]), sims[1:].tolist())
            inj_ndcg = _ndcg_from_rank(inj_rank)
            neg_above_pos = bool(np.any(sims[1:] > sims[0]))

            rows.append(dict(
                task="queries",
                qid=ex.qid,
                base_query=ex.query,
                identity_entity=ent, identity_label=lab, identity_category=cat, identity_display=disp,
                where=where,
                base_rank=base_rank, inj_rank=inj_rank, rank_delta=(inj_rank - base_rank),
                base_ndcg=base_ndcg, inj_ndcg=inj_ndcg, ndcg_delta=(inj_ndcg - base_ndcg),
                neg_above_pos=int(neg_above_pos),
                order_unchanged=int(base_order == _argsort_desc(sims.tolist())),
                q_self_drift=float(drift_all[row_i]),
                hard_match=int(inj_rank == base_rank),
                soft_match=int(inj_rank <= topk),
            ))

    return pd.DataFrame(rows)



# ====================================================
# Retrieval eval (passages)
# ====================================================


def evaluate_retrieval_passages(
    items: List[NQItem],
    embed_model: SentenceTransformer,
    pairs: List[Tuple[str,str,str]],
    begin_and_end: bool = True,
    topk: int = 3,
    progress: bool = True,
    embed_batch_size: int = 64,
) -> pd.DataFrame:
    rows = []
    it = tqdm(items, desc="RQ3: Retrieval (passages)", disable=not progress)

    for ex in it:
        # Embed base query once
        q_vec = batch_embed_any([ex.query], embed_model, batch_size=embed_batch_size, progress=False)[0]

        # Embed base docs once: [pos_base] + negs
        base_docs  = [ex.pos_doc] + list(ex.neg_docs)
        base_dvecs = batch_embed_any(base_docs, embed_model, batch_size=embed_batch_size, progress=False)
        pos_base_vec = base_dvecs[0]
        neg_mat = base_dvecs[1:]                      # shape: (Nneg, d)

        # Precompute sims of negs to q (reuse for all injections)
        sims_negs = (neg_mat @ q_vec) if neg_mat.size else np.array([], dtype=np.float32)
        sim_pos_base = float(pos_base_vec @ q_vec)

        # Keep only if base positive is strictly top-1
        if not (sims_negs.size == 0 or (sim_pos_base > np.max(sims_negs))):
            continue

        base_rank = _rank_of_positive(sim_pos_base, sims_negs.tolist())
        base_ndcg = _ndcg_from_rank(base_rank)

        # Iterate injections (only re-embed the injected positive)
        where_list = (["begin","end"] if begin_and_end else ["begin"])
        for where in where_list:
            for ent, lab, cat in pairs:
                disp, _ = realize_identity_ref(ent, lab, cat)
                pos_inj = inject_passage(ex.pos_doc, ent, lab, cat, where)

                pos_inj_vec = batch_embed_any([pos_inj], embed_model, batch_size=embed_batch_size, progress=False)[0]
                drift = 1.0 - float(pos_base_vec @ pos_inj_vec)

                sim_inj_pos = float(pos_inj_vec @ q_vec)
                inj_rank = _rank_of_positive(sim_inj_pos, sims_negs.tolist())
                inj_ndcg = _ndcg_from_rank(inj_rank)
                neg_above_pos = bool(np.any(sims_negs > sim_inj_pos))

                rows.append(dict(
                    task="passages",
                    qid=ex.qid,
                    base_query=ex.query,
                    identity_entity=ent, identity_label=lab, identity_category=cat, identity_display=disp,
                    where=where,
                    base_rank=base_rank, inj_rank=inj_rank, rank_delta=(inj_rank - base_rank),
                    base_ndcg=base_ndcg, inj_ndcg=inj_ndcg, ndcg_delta=(inj_ndcg - base_ndcg),
                    neg_above_pos=int(neg_above_pos),
                    p_self_drift=drift,
                    hard_match=int(inj_rank == base_rank),
                    soft_match=int(inj_rank <= topk),
                ))
    return pd.DataFrame(rows)


# ====================================================
# filters and normalization
# ====================================================
def _normalize_answer(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()

def _string_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def _answer_consistency_flags(base: str, inj: str, pos_doc: Optional[str] = None, thr_same: float = 0.8) -> Dict[str, Any]:
    # compute string-level text similarity 
    # they tell how the generation changed at the surface level (before we look at meaning).
    nb, ni = _normalize_answer(base), _normalize_answer(inj)
    sim = _string_sim(nb, ni)

    exact = 1 if (nb == ni and nb != "") else 0 #normalized strings are identical and non-empty.
    near  = 1 if sim >= thr_same else 0 # difflib ratio ≥ 0.8 (tunable). “close enough” wording.

    # IMPORTANT: force booleans; don't rely on `and`/`or` short-circuit values
    contains_bool = (bool(nb) and (nb in ni)) or (bool(ni) and (ni in nb)) # one normalized string contains the other (e.g., “Eiffel Tower” vs “the Eiffel Tower in Paris”).
    contain = 1 if contains_bool else 0

    in_pos = np.nan
    if pos_doc:
        nd = _normalize_answer(pos_doc)
        both_bool = (bool(nb) and (nb in nd)) and (bool(ni) and (ni in nd)) # both the base and injected answers appear as substrings in the positive passage (after normalization) → sanity check that both are grounded by the passage.
        in_pos = 1 if both_bool else 0

    return dict(
        exact_match=exact,
        near_match=near,
        contains=contain,
        both_supported=in_pos,
        str_sim=sim, # the actual difflib similarity score (0–1).
    )


# ====================================================
# Generation via RITS
# ====================================================

def evaluate_generation(
    items: List[NQItem],
    rits: Optional[RITSClient],
    gen_model_name: str,
    pairs: List[Tuple[str,str,str]],
    begin_and_end: bool = True,
    max_tokens: int = 120,
    temperature: float = 0.2,
    progress: bool = True,
    embed_model: Optional[SentenceTransformer] = None,
    gen_batch_size: int = 64,
    embed_batch_size: int = 64,
) -> pd.DataFrame:
    if rits is None: # If rits is None → returns empty DF (generation is optional).
        return pd.DataFrame()  # skip neatly

    rows, prompts, keys = [], [], []

    for ex in items: # Builds a list of prompts:
        prompts.append(ex.query)
        keys.append((ex.qid, "base", "", "", "", ex.query, ex.pos_doc)) # For each qid: one base prompt (the raw query),
        for where in (["begin","end"] if begin_and_end else ["begin"]): 
            for ent, lab, cat in pairs:
                q_inj = inject_query(ex.query, ent, lab, cat, where)
                prompts.append(q_inj)
                keys.append((ex.qid, where, ent, lab, cat, ex.query, ex.pos_doc)) # For each (where, pair): an injected prompt (begin/end).

    outs = rits.batched_generate(
        gen_model_name, prompts, max_tokens=max_tokens, temperature=temperature,
        batch_size=gen_batch_size, progress=progress)

    # map base completions per qid
    base_map: Dict[str, str] = {} # Associates injected completion with its base (per qid) 
    for (qid, where, *_), text in zip(keys, outs):
        if where == "base":
            base_map[qid] = text

    for (qid, where, ent, lab, cat, _q, pos_doc), text in zip(keys, outs): # computes Text similarity flags (exact/near/contains/both_supported/str_sim),
        if where == "base":
            continue
        base_text = base_map.get(qid, "")
        flags = _answer_consistency_flags(base_text, text, pos_doc)
        rows.append(dict(
            qid=qid, where=where,
            identity_entity=ent, identity_label=lab, identity_category=cat,
            identity_display=realize_identity_ref(ent, lab, cat)[0],
            base_completion=base_text, inj_completion=text,
            str_sim=flags["str_sim"],
            exact_match=flags["exact_match"],
            near_match=flags["near_match"],
            contains=flags["contains"],
            both_supported=(flags["both_supported"] if flags["both_supported"] is not None else np.nan),
        )
        )
    return pd.DataFrame(rows)

# ====================================================
# Aggregates & tests
# ====================================================
def agg_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    g = (df.groupby(["identity_display","identity_category","where"], as_index=False)
           .agg(mean_rank_delta=("rank_delta","mean"),
                sd_rank_delta=("rank_delta","std"),
                hard_match_rate=("hard_match","mean"),
                soft_match_rate=("soft_match","mean"),
                n=("rank_delta","count")))
    return g.sort_values(["identity_category","where","mean_rank_delta"])

def between_group_ols_anova(df_rows: pd.DataFrame, metric: str = "rank_delta"):
    data = df_rows[df_rows["where"].isin(["begin","end"])].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric, "identity_category", "where"])
    if data.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    model = smf.ols(f"{metric} ~ C(identity_category) + C(where) + C(identity_category):C(where)",
                    data=data).fit(cov_type="HC3")
    aov = anova_lm(model, typ=2)
    cell_means = (data.groupby(["identity_category","where"], as_index=False)
                        .agg(mean_delta=(metric,"mean"),
                             sd=(metric,"std"),
                             n=(metric,"count")))
    return model, aov, cell_means

def begin_end_diff_by_category(df_rows: pd.DataFrame, metric: str = "rank_delta"):
    sub = df_rows[df_rows["where"].isin(["begin","end"])].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    # columns we always need
    must_have = {"qid", "identity_category", "where", metric}
    if not must_have.issubset(sub.columns):
        return pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame()

    # Optional: identity_display (present in retrieval, absent in generation)
    group_keys = ["qid", "identity_category", "where"]
    if "identity_display" in sub.columns:
        group_keys.insert(1, "identity_display")

    sub = sub.dropna(subset=list(must_have))
    if sub.empty:
        return pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame()

    piv = sub.groupby(group_keys)[metric].mean().unstack("where")
    piv = piv.dropna(subset=["begin","end"])
    if piv.empty:
        return pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame()

    piv["diff"] = piv["begin"] - piv["end"]
    dat = piv.reset_index()
    mod = smf.ols("diff ~ C(identity_category)", data=dat).fit(cov_type="HC3")
    aov = anova_lm(mod, typ=2)
    sums = (dat.groupby("identity_category", as_index=False)
                .agg(mean_diff=("diff","mean"), sd=("diff","std"), n=("diff","count")))
    return dat, mod, aov, sums


def gen_drift_long(gen_cos_rows: pd.DataFrame) -> pd.DataFrame:
    if gen_cos_rows is None or gen_cos_rows.empty:
        return pd.DataFrame(columns=["qid","identity_category","where","drift"])
    parts = []
    for pos, col in [("begin","cos_base_begin"), ("end","cos_base_end")]:
        t = gen_cos_rows[["qid","identity_category", col]].copy()
        t["where"] = pos
        t["drift"] = 1.0 - pd.to_numeric(t[col], errors="coerce")
        parts.append(t[["qid","identity_category","where","drift"]])
    return pd.concat(parts, ignore_index=True)



# ---------- Significance on arbitrary delta metric (rank_delta, ndcg_delta, etc.) ----------
def rq3_significance_by_category(df_rows: pd.DataFrame, metric: str = "rank_delta") -> pd.DataFrame:
    out = []
    for (cat, where), g in df_rows.groupby(["identity_category","where"]):
        x = pd.to_numeric(g.get(metric, np.nan), errors="coerce").to_numpy(float)
        x = x[~np.isnan(x)]
        if x.size < 2: 
            continue
        t, p = stats.ttest_1samp(x, 0.0)
        mean = float(np.mean(x)); sd = float(np.std(x, ddof=1)); n = int(x.size)
        se = sd / np.sqrt(n); ci = 1.96 * se
        out.append(dict(identity_category=cat, where=where, metric=metric,
                        n=n, mean_delta=mean, ci_low=mean-ci, ci_high=mean+ci,
                        t_stat=float(t), t_p=float(p)))
    return pd.DataFrame(out).sort_values(["identity_category","where","t_p"])


def rq3_significance_tables_metric(df_rows: pd.DataFrame, metric: str = "rank_delta") -> pd.DataFrame:
    """
    H0: mean(Δ_metric) = 0  (no effect of injection)
    Δ_metric can be 'rank_delta', 'ndcg_delta', or any column of deltas.
    H0: mean(Δ_metric) = 0  (no effect of injection)
    H1: mean(Δ_metric) ≠ 0
    Δ_metric is either rank_delta or ndcg_delta (or any Δ you add).

H0 (per identity & position): mean(rank_delta) = 0 (injection doesn’t change rank).
H1: mean(rank_delta) ≠ 0.
Negative mean Δ with small p → injection worsens ranking (pos doc moves down).
Positive mean Δ with small p → injection improves ranking.
 running the same tests separately for queries (identity in query) and passages (identity in pos doc). That answers:
“Does putting identity in the query shift retrieval?”
“Does putting identity in the doc shift retrieval?” """
    out = []
    for (disp, cat, where), g in df_rows.groupby(["identity_display","identity_category","where"]):
        x = pd.to_numeric(g.get(metric, np.nan), errors="coerce").to_numpy(float)
        x = x[~np.isnan(x)]
        if x.size < 2:
            continue
        t, p = stats.ttest_1samp(x, 0.0)
        mean = float(np.mean(x)); sd = float(np.std(x, ddof=1)); n = int(x.size)
        se = sd / np.sqrt(n); ci = 1.96 * se
        out.append(dict(
            identity_display=disp,
            identity_category=cat,
            where=where,
            metric=metric,
            n=n,
            mean_delta=mean,
            ci_low=mean-ci,
            ci_high=mean+ci,
            t_stat=float(t),
            t_p=float(p),
        ))
    return pd.DataFrame(out).sort_values(["identity_category","where","t_p"])


def generation_drift_ttests(gen_cos_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Uses rq3_generation_cosine_rows.csv schema:
    columns include: qid, identity_category, cos_base_begin, cos_base_end
    H0: mean(drift) = 0 per (category, position), where drift = 1 - cos(base, injected).
    """
    out = []
    if gen_cos_rows is None or gen_cos_rows.empty:
        return pd.DataFrame(columns=["identity_category","position","n","mean_drift","ci_low","ci_high","t_stat","t_p"])
    for cat, g in gen_cos_rows.groupby("identity_category"):
        for pos, col in [("begin","cos_base_begin"), ("end","cos_base_end")]:
            x = 1.0 - pd.to_numeric(g.get(col, np.nan), errors="coerce").to_numpy(float)
            x = x[~np.isnan(x)]
            if x.size < 2:
                continue
            t, p = stats.ttest_1samp(x, 0.0)
            mean = float(np.mean(x)); sd = float(np.std(x, ddof=1)); n = int(x.size)
            se = sd / np.sqrt(n); ci = 1.96 * se
            out.append(dict(
                identity_category=cat, position=pos, n=n,
                mean_drift=mean, ci_low=mean-ci, ci_high=mean+ci,
                t_stat=float(t), t_p=float(p)
            ))
    return pd.DataFrame(out).sort_values(["identity_category","position"])


def generation_drift_agg(gen_cos_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates mean drift (1 - cos) with 95% CI per category × position.
    Returns columns like the cosine-agg but for drift:
      identity_category, mean_begin, mean_end, n, ci_begin_lo/hi, ci_end_lo/hi
    """
    if gen_cos_rows is None or gen_cos_rows.empty:
        return pd.DataFrame(columns=[
            "identity_category","mean_begin","mean_end","n",
            "ci_begin_lo","ci_begin_hi","ci_end_lo","ci_end_hi"
        ])

    rows = []
    for cat, g in gen_cos_rows.groupby("identity_category"):
        b = 1.0 - pd.to_numeric(g.get("cos_base_begin", np.nan), errors="coerce").to_numpy(float)
        e = 1.0 - pd.to_numeric(g.get("cos_base_end",   np.nan), errors="coerce").to_numpy(float)
        b = b[~np.isnan(b)]; e = e[~np.isnan(e)]
        mb = float(np.mean(b)) if b.size else np.nan
        me = float(np.mean(e)) if e.size else np.nan
        nb = int(b.size); ne = int(e.size)
        cib = 1.96 * (np.std(b, ddof=1) / np.sqrt(nb)) if nb > 1 else np.nan
        cie = 1.96 * (np.std(e, ddof=1) / np.sqrt(ne)) if ne > 1 else np.nan
        rows.append(dict(
            identity_category=cat, mean_begin=mb, mean_end=me,
            n=min(nb, ne) if (nb and ne) else max(nb, ne),
            ci_begin_lo=mb - cib if nb > 1 else np.nan,
            ci_begin_hi=mb + cib if nb > 1 else np.nan,
            ci_end_lo=me - cie if ne > 1 else np.nan,
            ci_end_hi=me + cie if ne > 1 else np.nan,
        ))
    return pd.DataFrame(rows)


def plot_generation_drift(agg_df: pd.DataFrame, out_path: str):
    """
    Bar plot of mean drift (1 - cos) by category with 95% CI for begin vs end.
    """
    if agg_df is None or agg_df.empty:
        print(f"[PLOT:GEN_DRIFT] nothing to plot for {out_path}")
        return
    x = np.arange(len(agg_df))
    w = 0.38
    eb_begin = [agg_df["mean_begin"] - agg_df["ci_begin_lo"], agg_df["ci_begin_hi"] - agg_df["mean_begin"]]
    eb_end   = [agg_df["mean_end"]   - agg_df["ci_end_lo"],   agg_df["ci_end_hi"]   - agg_df["mean_end"]]

    plt.figure(figsize=(14, 6))
    plt.bar(x - w/2, agg_df["mean_begin"], width=w, label="begin", yerr=eb_begin, capsize=3, alpha=0.9)
    plt.bar(x + w/2, agg_df["mean_end"],   width=w, label="end",   yerr=eb_end,   capsize=3, alpha=0.9)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, agg_df["identity_category"], rotation=45, ha="right")
    plt.ylabel("Generation drift (1 − cos)")
    plt.title("RQ3 Generation Drift by Category (mean ±95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[PLOT:GEN_DRIFT] wrote → {out_path}")

def _ci95_from_std(std, n):
    std = np.asarray(std, float)
    n = np.asarray(n, float)
    se = std / np.sqrt(np.clip(n, 1, None))
    return 1.96 * se


def rq3_generation_cosine_rows(gen_df: pd.DataFrame, embed_model, batch_size: int = 64) -> pd.DataFrame:
    """
    Build per-qid cosine(base_completion, inj_completion) for begin/end.
    Returns columns: qid, identity_category, cos_base_begin, cos_base_end.
    """
    # rq3_generation_cosine_rows: embeds base vs injected completions and computes cosine(base, injected) per QID and position. 
    if gen_df is None or gen_df.empty: 
        return pd.DataFrame(columns=["qid","identity_category","cos_base_begin","cos_base_end"])

    # Keep only rows that have both texts and a valid where
    sub = gen_df.dropna(subset=["base_completion","inj_completion","where"]).copy()
    sub = sub[sub["where"].isin(["begin","end"])]
    if sub.empty:
        return pd.DataFrame(columns=["qid","identity_category","cos_base_begin","cos_base_end"])

    # Batch-embed (base, inj) pairs
    texts = []
    for _, r in sub.iterrows():
        texts.append(str(r["base_completion"]))
        texts.append(str(r["inj_completion"]))

    vecs = batch_embed_any(texts, embed_model, normalize=True, batch_size=batch_size,
                           progress=True, desc="RQ3: embed (gen base+inj)")

    # Compute pairwise cosines
    cos = []
    for i in range(0, len(vecs), 2):
        v_base = vecs[i]
        v_inj  = vecs[i+1]
        cos.append(cos_sim_single(v_base, v_inj))

    sub = sub.reset_index(drop=True)
    sub["cos_base_inj"] = cos

    # Average per (qid, category, where), then pivot to columns begin/end
    gb = (sub.groupby(["qid","identity_category","where"], as_index=False)
              .agg(cos_base_inj=("cos_base_inj","mean")))
    piv = gb.pivot(index=["qid","identity_category"], columns="where", values="cos_base_inj").reset_index()
    # Ensure both columns exist
    if "begin" not in piv.columns: piv["begin"] = np.nan
    if "end"   not in piv.columns: piv["end"]   = np.nan
    piv = piv.rename(columns={"begin":"cos_base_begin", "end":"cos_base_end"})
    return piv[["qid","identity_category","cos_base_begin","cos_base_end"]]



def rq3_generation_cosine_agg(cos_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cosine rows by identity_category; report means and 95% CI.
    Expects columns: qid, identity_category, cos_base_begin, cos_base_end.
    """

    if cos_rows is None or cos_rows.empty:
        return pd.DataFrame(columns=[
            "identity_category","mean_begin","mean_end","n",
            "ci_begin_lo","ci_begin_hi","ci_end_lo","ci_end_hi"
        ])

    # Keep only needed columns and coerce to float
    cols = ["identity_category","cos_base_begin","cos_base_end"]
    cos_rows = cos_rows[cols].copy()
    cos_rows["cos_base_begin"] = pd.to_numeric(cos_rows["cos_base_begin"], errors="coerce")
    cos_rows["cos_base_end"]   = pd.to_numeric(cos_rows["cos_base_end"], errors="coerce")

    def _ci(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if x.size < 2:
            return np.nan
        se = np.std(x, ddof=1) / np.sqrt(x.size)
        return 1.96 * se

    grp = cos_rows.groupby("identity_category", dropna=False)
    out = grp.agg(
        mean_begin=("cos_base_begin","mean"),
        mean_end=("cos_base_end","mean"),
        n=("cos_base_begin","count")
    ).reset_index()

    # 95% CI half-widths
    w_begin = grp["cos_base_begin"].apply(_ci).values
    w_end   = grp["cos_base_end"].apply(_ci).values
    out["ci_begin_lo"] = out["mean_begin"] - w_begin
    out["ci_begin_hi"] = out["mean_begin"] + w_begin
    out["ci_end_lo"]   = out["mean_end"]   - w_end
    out["ci_end_hi"]   = out["mean_end"]   + w_end
    return out

# ====================================================
# Plots
# ====================================================
def plot_rq3_significance(sig_q_path, sig_p_path, out_path="rq3_rankdelta.png"):
    q = pd.read_csv(sig_q_path)
    p = pd.read_csv(sig_p_path)

    # Pick a usable mean column (rank code wrote mean_delta; older code had mean_rank_delta)
    col_q = "mean_rank_delta" if "mean_rank_delta" in q.columns else ("mean_delta" if "mean_delta" in q.columns else None)
    col_p = "mean_rank_delta" if "mean_rank_delta" in p.columns else ("mean_delta" if "mean_delta" in p.columns else None)
    if col_q is None or col_p is None:
        print("[PLOT] missing mean columns; skipping plot")
        return

    q_means = q.groupby("identity_category")[col_q].mean()
    p_means = p.groupby("identity_category")[col_p].mean()

    categories = sorted(set(q_means.index) | set(p_means.index))
    queries_vals = [q_means.get(cat, 0.0) for cat in categories]
    passages_vals = [p_means.get(cat, 0.0) for cat in categories]

    width = 0.4
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x - width/2, queries_vals, width, label="Queries")
    ax.bar(x + width/2, passages_vals, width, label="Passages")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Mean Rank Δ (inj − base)")
    ax.set_title("RQ3: Retrieval Drift by Identity Category")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] saved → {out_path}")


def plot_rq3_retrieval_by_category(df_rows_path: str, out_path: str, title: str):
    df = pd.read_csv(df_rows_path)
    sub = df[df["where"].isin(["begin","end"])].copy()
    if sub.empty:
        print(f"[PLOT:RQ3:RET] no rows in {df_rows_path}")
        return

    gb = (sub.groupby(["identity_category","where"], as_index=False)
             .agg(mean_rank_delta=("rank_delta","mean"),
                  std=("rank_delta","std"),
                  n=("rank_delta","count")))
    gb["ci"] = _ci95_from_std(gb["std"], gb["n"])

    piv = gb.pivot(index="identity_category", columns="where", values="mean_rank_delta")
    ci_piv = gb.pivot(index="identity_category", columns="where", values="ci")
    for col in ["begin","end"]:
        if col not in piv.columns: piv[col] = float("nan")
        if col not in ci_piv.columns: ci_piv[col] = float("nan")

    piv = piv[["begin","end"]].sort_values("begin")
    ci_piv = ci_piv.loc[piv.index, ["begin","end"]]

    cats = list(piv.index)
    x = range(len(cats))
    width = 0.42

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar([i - width/2 for i in x], piv["begin"], width,
           yerr=ci_piv["begin"], capsize=3, label="begin")
    ax.bar([i + width/2 for i in x], piv["end"], width,
           yerr=ci_piv["end"], capsize=3, label="end")
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Mean Rank Δ (inj − base)")
    ax.set_title(title + " (mean ±95% CI)")
    ax.legend()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT:RQ3:RET] wrote → {out_path}")

# ============================================
# Experiment Data Accounting Utilities
# ============================================
def summarize_rq3(results: Dict[str, pd.DataFrame]):
    q_df = results.get("rq3_retrieval_queries_row")
    p_df = results.get("rq3_retrieval_passages_row")
    g_df = results.get("rq3_generation_row")

    n_identities = None
    if q_df is not None and not q_df.empty:
        n_identities = q_df["identity_display"].nunique()

    n_queries  = q_df["qid"].nunique() if (q_df is not None and not q_df.empty) else None
    n_passages = p_df["qid"].nunique() if (p_df is not None and not p_df.empty) else None
    n_gen = len(g_df) if (g_df is not None and not g_df.empty) else 0

    return (f"RQ3 → identities={n_identities}, queries={n_queries}, "
            f"passages={n_passages}, gen_rows={n_gen}")

# ====================================================
# Pipeline
# ====================================================
def run_rq3_pipeline(
    granite_df: pd.DataFrame,
    nq_jsonl_path: str,
    embed_model: SentenceTransformer,
    rits_api_key: Optional[str],
    out_dir: Optional[str] = None,
    gen_model_name: str = "ibm-granite/granite-13b-instruct-v2",
    sample_n_queries: int = 300,
    per_category: int = 6,
    begin_and_end: bool = True,
    topk: int = 3,
    max_gen_tokens: int = 120,
    temperature: float = 0.2,
    progress: bool = True,
    embed_batch_size: int = 64,
    gen_batch_size: int = 64,
) -> Dict[str, pd.DataFrame]:
    pairs = harvest_bbq_pairs_from_granite(
        granite_df, per_category=per_category, max_pairs=None,
        shuffle_within_cat=True, seed=SEED, progress=progress
    )
    cats = sorted({c for _,_,c in pairs})
    print(f"[RQ3] identities: {len(pairs)} across {len(cats)} categories")

    items = load_nq_pairs_jsonl(nq_jsonl_path, max_items=sample_n_queries)
    print(f"[RQ3] loaded NQ pairs: {len(items)} from {nq_jsonl_path}")

    df_q = evaluate_retrieval_queries(items, embed_model, pairs, begin_and_end=begin_and_end, topk=topk, progress=progress)
    df_p = evaluate_retrieval_passages(items, embed_model, pairs, begin_and_end=begin_and_end, topk=topk, progress=progress)


    rits = None
    if rits_api_key:
        try:
            rits = RITSClient(api_key=rits_api_key)
        except Exception as e:
            print(f"[RQ3] RITS disabled → {e}")

    df_g = evaluate_generation(
        items, rits, gen_model_name, pairs,
        begin_and_end=begin_and_end, max_tokens=max_gen_tokens,
        temperature=temperature, progress=progress,
        embed_model=embed_model,
        gen_batch_size=gen_batch_size,
        embed_batch_size=embed_batch_size,
    )

    agg_q = agg_retrieval(df_q)
    agg_p = agg_retrieval(df_p)

    return {
        "rq3_retrieval_queries_row": df_q,
        "rq3_retrieval_passages_row": df_p,
        "rq3_retrieval_queries_agg": agg_q,
        "rq3_retrieval_passages_agg": agg_p,
        "rq3_generation_row": df_g,
    }

# ====================================================
# Main / CLI
# ====================================================
def main():
    ap = argparse.ArgumentParser()

    # Shared
    ap.add_argument("--model", default="ibm-granite/granite-embedding-30m-english")
    ap.add_argument("--out-dir", default="./out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen-batch-size", type=int, default=64,
                    help="Batch size for RITS generation batching")

    # RQ3 controls
    ap.add_argument("--rq3-nq-jsonl", type=str, default="NQ-train_pairs_train_all.jsonl")
    ap.add_argument("--rq3-sample", type=int, default=300)
    ap.add_argument("--rq3-per-category", type=int, default=6)
    ap.add_argument("--rq3-topk", type=int, default=3)
    # Toggle: default = both begin & end; set this flag to use begin-only
    ap.add_argument("--rq3-only-begin", action="store_true",
                    help="Inject identity at begin only (default: both begin and end)")
    ap.add_argument("--rq3-gen-model", type=str, default="ibm-granite/granite-13b-instruct-v2")
    ap.add_argument("--rq3-gen-tokens", type=int, default=120)
    ap.add_argument("--rq3-rits-key", type=str, default=None) # (No --rq3-rits-key → generation is skipped; you’ll still get all retrieval stats/plots.)

    # Embedding backend
    ap.add_argument("--use-rits-embed", action="store_true",
                    help="Use RITS embeddings instead of sentence-transformers")
    ap.add_argument("--rits-key", type=str, default=os.getenv("RITS_API_KEY"),
                    help="RITS API key (or set env RITS_API_KEY)")
    ap.add_argument("--rits-embed-model", type=str, default="ibm/slate-125m-english-rtrvr-v2",
                    help="RITS embedding model name")
    ap.add_argument("--embed-batch-size", type=int, default=64,
                    help="Batch size for embedding calls")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load granite.* from prior stage
    granite = load_granite(args.out_dir)
    print("[LOAD] granite dtypes:", granite.dtypes.to_dict())
    print(f"[LOAD] Loaded existing granite.csv/parquet → {granite.shape}")

    # Embedder (ST or RITS)
    model = load_embedder_from_args(args)

    # Determine injection mode
    begin_and_end = not args.rq3_only_begin  # True = both positions; False = begin-only
    print(f"[RQ3] injection positions: {'begin+end' if begin_and_end else 'begin-only'}")

    # ---------------- RQ3 ----------------
    print("[RQ3] starting...")
    rq3 = run_rq3_pipeline(
        granite_df=granite,
        nq_jsonl_path=args.rq3_nq_jsonl,
        embed_model=model,
        rits_api_key=args.rq3_rits_key,
        gen_model_name=args.rq3_gen_model,
        out_dir=args.out_dir,
        sample_n_queries=args.rq3_sample,
        per_category=args.rq3_per_category, # per_category limits how many identities keep per category in the global pool of candidate identities (pairs) After that, for each question, we sample pairs_per_question identities from the entire global pool: local_pairs = pick_pairs_for_question(pairs, pairs_per_question, rng) That’s a random draw across all categories combined, not per category.
        begin_and_end=begin_and_end,
        topk=args.rq3_topk,
        max_gen_tokens=args.rq3_gen_tokens,
        temperature=0.2,
        progress=True,
        embed_batch_size=args.embed_batch_size,
        gen_batch_size=args.gen_batch_size,
    )


    # ---------- RQ3 significance & save ----------
    out_dir = args.out_dir
    df_q_rows = rq3.get("rq3_retrieval_queries_row", pd.DataFrame())
    df_p_rows = rq3.get("rq3_retrieval_passages_row", pd.DataFrame())

    # Primary: per-identity one-sample t-tests on rank and NDCG deltas
    sig_rank_q = rq3_significance_tables_metric(df_q_rows, metric="rank_delta")
    sig_rank_p = rq3_significance_tables_metric(df_p_rows, metric="rank_delta")
    sig_ndcg_q = rq3_significance_tables_metric(df_q_rows, metric="ndcg_delta")
    sig_ndcg_p = rq3_significance_tables_metric(df_p_rows, metric="ndcg_delta")

    sig_rank_q.to_csv(os.path.join(out_dir, "rq3_significance_queries_rank.csv"), index=False)
    sig_rank_p.to_csv(os.path.join(out_dir, "rq3_significance_passages_rank.csv"), index=False)
    sig_ndcg_q.to_csv(os.path.join(out_dir, "rq3_significance_queries_ndcg.csv"), index=False)
    sig_ndcg_p.to_csv(os.path.join(out_dir, "rq3_significance_passages_ndcg.csv"), index=False)

    # ---------- Also: pooled-by-category versions (same tests, drop identity_display) ----------
    sig_rank_q_cat = rq3_significance_by_category(df_q_rows, metric="rank_delta")
    sig_rank_p_cat = rq3_significance_by_category(df_p_rows, metric="rank_delta")
    sig_ndcg_q_cat = rq3_significance_by_category(df_q_rows, metric="ndcg_delta")
    sig_ndcg_p_cat = rq3_significance_by_category(df_p_rows, metric="ndcg_delta")

    sig_rank_q_cat.to_csv(os.path.join(out_dir, "rq3_sig_queries_rank_bycat.csv"), index=False) #potential upgrade.. Add significance stars to bars using the p-values from *_bycat.csv files.
    sig_rank_p_cat.to_csv(os.path.join(out_dir, "rq3_sig_passages_rank_bycat.csv"), index=False)
    sig_ndcg_q_cat.to_csv(os.path.join(out_dir, "rq3_sig_queries_ndcg_bycat.csv"), index=False)
    sig_ndcg_p_cat.to_csv(os.path.join(out_dir, "rq3_sig_passages_ndcg_bycat.csv"), index=False)

    # -------- Generation (Option A drift = 1 - cos) --------
    gen = rq3.get("rq3_generation_row", pd.DataFrame())
    gen_cos_rows = pd.DataFrame()
    gen_cos_agg  = pd.DataFrame()
    gen_drift_sig = pd.DataFrame()
    gen_drift_agg = pd.DataFrame()

    if isinstance(gen, pd.DataFrame) and not gen.empty:
        # ⬇️ use your CLI-provided embed batch size here
        gen_cos_rows = rq3_generation_cosine_rows(gen, model, batch_size=args.embed_batch_size)
        gen_cos_rows.to_csv(os.path.join(out_dir, "rq3_generation_cosine_rows.csv"), index=False)

        # Aggregates (cosine and drift) + one-sample t-tests on drift
        gen_cos_agg  = rq3_generation_cosine_agg(gen_cos_rows)
        gen_drift_sig = generation_drift_ttests(gen_cos_rows)
        gen_drift_agg = generation_drift_agg(gen_cos_rows)

        if not gen_cos_agg.empty:
            gen_cos_agg.to_csv(os.path.join(out_dir, "rq3_generation_cosine_agg.csv"), index=False)
        if not gen_drift_sig.empty:
            gen_drift_sig.to_csv(os.path.join(out_dir, "rq3_significance_generation_drift.csv"), index=False)
        if not gen_drift_agg.empty:
            gen_drift_agg.to_csv(os.path.join(out_dir, "rq3_generation_drift_agg.csv"), index=False)


    # Save all raw tables from pipeline
    for key, df_out in rq3.items():
        if isinstance(df_out, pd.DataFrame) and not df_out.empty:
            path = os.path.join(out_dir, f"{key}.csv")
            df_out.to_csv(path, index=False)
            print(f"[SAVE] {key} → {path} (rows={len(df_out)})")

    # -------- Between-group ANOVA / paired begin–end (optional but helpful) --------
    # may want to upgrade and add a difference plot (begin−end) per category for queries and for passages; that maps directly to the paired begin–end ANOVA tables
    try:
        # Retrieval – queries
        if not df_q_rows.empty:
            mod_q, aov_q, cells_q = between_group_ols_anova(df_q_rows, metric="rank_delta")
            if aov_q is not None and not aov_q.empty:
                print("\n[ANOVA] Queries (rank_delta)\n", aov_q)
                print(cells_q.sort_values(["identity_category","where"]))
                aov_q.to_csv(os.path.join(out_dir, "rq3_anova_queries_rank.csv"))
                cells_q.to_csv(os.path.join(out_dir, "rq3_anova_queries_rank_cells.csv"), index=False)
            dat_q, mod_q2, aov_q2, sums_q = begin_end_diff_by_category(df_q_rows, metric="rank_delta")
            if aov_q2 is not None and not aov_q2.empty:
                print("\n[ANOVA] Paired begin−end (Queries, rank_delta)\n", aov_q2)
                print(sums_q.sort_values("mean_diff", ascending=False))
                aov_q2.to_csv(os.path.join(out_dir, "rq3_anova_queries_begin_end.csv"))
                sums_q.to_csv(os.path.join(out_dir, "rq3_anova_queries_begin_end_summ.csv"), index=False)

        # Retrieval – passages
        if not df_p_rows.empty:
            mod_p, aov_p, cells_p = between_group_ols_anova(df_p_rows, metric="rank_delta")
            if aov_p is not None and not aov_p.empty:
                print("\n[ANOVA] Passages (rank_delta)\n", aov_p)
                print(cells_p.sort_values(["identity_category","where"]))
                aov_p.to_csv(os.path.join(out_dir, "rq3_anova_passages_rank.csv"))
                cells_p.to_csv(os.path.join(out_dir, "rq3_anova_passages_rank_cells.csv"), index=False)
            dat_p, mod_p2, aov_p2, sums_p = begin_end_diff_by_category(df_p_rows, metric="rank_delta")
            if aov_p2 is not None and not aov_p2.empty:
                print("\n[ANOVA] Paired begin−end (Passages, rank_delta)\n", aov_p2)
                print(sums_p.sort_values("mean_diff", ascending=False))
                aov_p2.to_csv(os.path.join(out_dir, "rq3_anova_passages_begin_end.csv"))
                sums_p.to_csv(os.path.join(out_dir, "rq3_anova_passages_begin_end_summ.csv"), index=False)

        # Generation – drift
        if not gen_cos_rows.empty:
            gen_long = gen_drift_long(gen_cos_rows)
            if not gen_long.empty:
                mod_g, aov_g, cells_g = between_group_ols_anova(
                    gen_long.rename(columns={"drift":"rank_delta"}), metric="rank_delta"
                )
                if aov_g is not None and not aov_g.empty:
                    print("\n[ANOVA] Generation drift (1 − cos)\n", aov_g)
                    print(cells_g.sort_values(["identity_category","where"]))
                    aov_g.to_csv(os.path.join(out_dir, "rq3_anova_generation_drift.csv"))
                    cells_g.to_csv(os.path.join(out_dir, "rq3_anova_generation_drift_cells.csv"), index=False)

                dat_g, mod_g2, aov_g2, sums_g = begin_end_diff_by_category(
                    gen_long.rename(columns={"drift":"rank_delta"}), metric="rank_delta"
                )
                if aov_g2 is not None and not aov_g2.empty:
                    print("\n[ANOVA] Paired begin−end (Generation drift)\n", aov_g2)
                    print(sums_g.sort_values("mean_diff", ascending=False))
                    aov_g2.to_csv(os.path.join(out_dir, "rq3_anova_generation_begin_end.csv"))
                    sums_g.to_csv(os.path.join(out_dir, "rq3_anova_generation_begin_end_summ.csv"), index=False)
    except Exception as e:
        print(f"[ANOVA] skipped → {e}")

    # ---------------- Plots ----------------
    try:
        # Retrieval drift overview (per-identity means)
        plot_rq3_significance(
            sig_q_path=os.path.join(out_dir, "rq3_significance_queries_rank.csv"),
            sig_p_path=os.path.join(out_dir, "rq3_significance_passages_rank.csv"),
            out_path=os.path.join(out_dir, "rq3_rankdelta.png")
        )
        # Category bar charts (mean ±95% CI) for queries/passages
        plot_rq3_retrieval_by_category(
            df_rows_path=os.path.join(out_dir, "rq3_retrieval_queries_row.csv"),
            out_path=os.path.join(out_dir, "rq3_queries_category_means.png"),
            title="RQ3 Retrieval (Queries): Rank Δ by Category"
        )
        plot_rq3_retrieval_by_category(
            df_rows_path=os.path.join(out_dir, "rq3_retrieval_passages_row.csv"),
            out_path=os.path.join(out_dir, "rq3_passages_category_means.png"),
            title="RQ3 Retrieval (Passages): Rank Δ by Category"
        )
        # Generation plots (cosine & drift) if available
        if not gen_cos_agg.empty:
            plot_rq3_generation_cosine(gen_cos_agg, out_path=os.path.join(out_dir, "rq3_generation_cosine.png"))
        if not gen_drift_agg.empty:
            plot_generation_drift(gen_drift_agg, out_path=os.path.join(out_dir, "rq3_generation_drift.png"))
    except Exception as e:
        print(f"[PLOT:RQ3] skipped → {e}")

    print(summarize_rq3(rq3))


if __name__ == "__main__":
    main()

