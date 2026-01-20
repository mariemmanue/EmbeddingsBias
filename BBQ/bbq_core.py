# bbq_core.py


import os
import math
import json
import re
import textwrap
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from datasets import load_dataset, get_dataset_config_names
from scipy import stats

# ----------------- SEEDING -----------------
SEED = int(os.environ.get("SEED", "123"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

VALID_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ----------------- DATA LOADING -----------------
def load_bbq_from_hf(dataset_id: str | None = None,
                     revision: str | None = None) -> pd.DataFrame:
    # Function to load a dataset from Hugging Face
    candidates = [dataset_id] if dataset_id else [
        "heegyu/BBQ",
        "Elfsong/BBQ",
        "walledai/BBQ",
    ]
    last_err = None
    for ds in candidates:
        if not ds:
            continue
        try:
            cfgs = get_dataset_config_names(ds)
            frames = []
            if not cfgs:
                dd = load_dataset(ds, revision=revision)
                for _, split in dd.items():
                    frames.append(split.to_pandas())
            else:
                for cfg in cfgs:
                    dd = load_dataset(ds, name=cfg, revision=revision)
                    for _, split in dd.items():
                        frames.append(split.to_pandas())
            df = pd.concat(frames, ignore_index=True)
            print(f"[HF] Loaded {len(df)} rows from '{ds}'")
            return df
        except Exception as e:
            last_err = e
            print(f"[WARN] Failed to load '{ds}': {e}")
    raise RuntimeError("Could not load BBQ from any mirror.") from last_err


def load_metadata(metadata_csv_path: str) -> pd.DataFrame:
    # Function to load metadata from a CSV file
    md = pd.read_csv(
        metadata_csv_path,
        keep_default_na=True,
        na_values=["NA", "Na", "na", ""],
        engine="python",
    )
    for col in ["category", "example_id", "question_index"]:
        if col in md.columns:
            md[col] = md[col].astype(str).str.strip()
    if "target_loc" not in md.columns:
        raise KeyError("Column 'target_loc' not found in additional_metadata.csv")
    md["target_loc"] = pd.to_numeric(md["target_loc"], errors="coerce").astype("Int64")
    return md


def choices_from_ans_fields(row: Dict[str, Any]) -> List[str]:
    out = []
    for k in ["ans0", "ans1", "ans2"]:
        if k in row and row[k] is not None:
            s = str(row[k]).strip()
            if s and s.lower() != "nan":
                out.append(s)
    if len(out) < 2:
        raise ValueError("Expected at least ans0 and ans1.")
    return out



def gold_letter_from_numeric_label(row: Dict[str, Any], n_choices: int) -> str:
# This function converts a numeric label into a corresponding letter. For example, 0 maps to A, 1 to B, etc.
    lab = row.get("label", None)
    if lab is None or (isinstance(lab, float) and np.isnan(lab)):
        return ""
    try:
        idx = int(lab)
        if 0 <= idx < n_choices:
            return VALID_LETTERS[idx]
        return ""
    except Exception:
        return str(lab).strip().upper()[:1]

# _UNKNOWN_PAT = re.compile(
#     r"(can't be determined|cannot be determined|undetermined|not determined|not answerable|cannot answer|"
#     r"not enough (?:info|information)|insufficient (?:info|information)|unknown|"
#     r"can't tell|cannot tell|can't answer|not sure|not known|unclear|impossible|not possible)",
#     re.I
# )

_UNKNOWN_PAT = re.compile(
    r"(can't be determined|cannot be determined|cant be determined|undetermined|"
    r"not determined|not answerable|cannot answer|can't answer|cant answer|"
    r"not enough ?(?:info|information)|insufficient ?(?:info|information)|"
    r"unknown|can't tell|cannot tell|cant tell|not sure|not known|unclear|"
    r"impossible|not possible)",
    re.I,
)


def _is_unknown_choice(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    return bool(_UNKNOWN_PAT.search(s))

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    # collapse whitespace + strip punctuation-ish edges
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r\"'.,;:!?()[]{}")
    return s

_DET_PREFIX = r"(?:the|a|an|my|your|his|her|their|our|this|that|these|those)"
_PREFIX_RE = re.compile(rf"^(?:{_DET_PREFIX})\s+", flags=re.I)

def _strip_det(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return _PREFIX_RE.sub("", s).strip()


def _anchor_tokens(ent: str) -> list[str]:
    """
    Last-resort tokens to look for in the context.
    Prefer proper-name anchors (TitleCase tokens).
    Else fall back to a few content-ish tokens.
    """
    ent = "" if ent is None else str(ent)

    # include digits + hyphens (e.g., "84-year-old")
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", ent)

    proper = [t for t in toks if t[0].isupper() and t.lower() not in {"the", "a", "an"}]
    if proper:
        return proper[:2]  # usually enough (first/last)

    stop = {
        "the","a","an","one","person","man","woman","guy","girl","boy","kid","individual",
        "from","with","who","whos","who's","is","was","are","were","on","in","of","to"
    }
    content = [t for t in toks if t.lower() not in stop and len(t) >= 3]
    return content[:3]

def _to_intish(x):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        if np.isnan(x):
            return None
        # only accept near-integers
        if float(x).is_integer():
            return int(x)
        return None
    if isinstance(x, str):
        xs = x.strip()
        xs = re.sub(r"\.0$", "", xs)  # "1.0" -> "1"
        if xs.isdigit():
            return int(xs)
    return None



def _first_match_pos(context: str, entity_text: str) -> int:
    """Return char start of (rough) entity mention in context, else -1."""
    if context is None or entity_text is None:
        return -1

    ctx = re.sub(r"\s+", " ", str(context)).strip()
    ent = re.sub(r"\s+", " ", str(entity_text)).strip()

    # normalize curly apostrophes
    ctx = ctx.replace("’", "'")
    ent = ent.replace("’", "'")

    if not ctx or not ent:
        return -1

    # 1) exact (case-insensitive)
    m = re.search(re.escape(ent), ctx, flags=re.I)
    if m:
        return m.start()

    # 2) if answer has comma, try head before comma (e.g., "Ayanna Dorsey, who's ...")
    if "," in ent:
        head = ent.split(",", 1)[0].strip()
        if head:
            m = re.search(rf"\b{re.escape(head)}\b", ctx, flags=re.I)
            if m:
                return m.start()

    # 3) normalize who's -> who is, and optionally drop "who is/was"
    ent_norm = ent.replace("who's", "who is")
    ent_norm = re.sub(r"\bwho\s+(is|was)\b", "", ent_norm, flags=re.I).strip()
    if ent_norm and ent_norm != ent:
        m = re.search(re.escape(ent_norm), ctx, flags=re.I)
        if m:
            return m.start()

    # 4) handle "The Muslim one" / "The gay one" etc.
    m_one = re.match(r"^(the\s+)?(.+?)\s+one$", ent, flags=re.I)
    if m_one:
        core = m_one.group(2).strip()
        pat = rf"\b{re.escape(core)}\b(?:\s+(?:one|man|woman|person|guy|girl|boy|kid|individual))?\b"
        m = re.search(pat, ctx, flags=re.I)
        if m:
            return m.start()

    # 5) determiner-strip logic
    ent2 = _strip_det(ent)
    if ent2 and ent2 != ent:
        pat = rf"\b(?:{_DET_PREFIX}\s+)?{re.escape(ent2)}\b"
        m2 = re.search(pat, ctx, flags=re.I)
        if m2:
            return m2.start()

        m3 = re.search(re.escape(ent2), ctx, flags=re.I)
        if m3:
            return m3.start()

    # 6) fallback: anchor token search (inside _first_match_pos)
    anchors = _anchor_tokens(ent)
    for a in anchors:
        for cand in {a, re.sub(r"('s|’s)$", "", a)}:
            if not cand:
                continue
            m4 = re.search(rf"\b{re.escape(cand)}\b", ctx, flags=re.I)
            if m4:
                return m4.start()


    return -1



def add_target_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - target_entity_text: the entity string corresponding to target_loc
      - other_entity_text: the non-target entity string (of the two entities)
      - target_char_start / other_char_start: first-match char index in context (debug)
      - target_position: {"first","second","unknown"}

    Assumptions:
      - Two "entity" answer options + one unknown option (order varies; we detect unknown by text)
      - target_loc is usually 0/1 (index into the two entity options) OR
        sometimes a letter ("A"/"B"/"C") OR
        sometimes an int index into ans0/ans1/ans2.
    """
    out = df.copy()

    # Ensure ans fields exist as strings
    for c in ["ans0", "ans1", "ans2"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
        else:
            out[c] = ""


    def _row_target_other(row) -> tuple[str, str]:
        context = row.get("context", "")

        # raw choices
        choices = [row.get("ans0", ""), row.get("ans1", ""), row.get("ans2", "")]
        choices = [
            c for c in choices
            if (c is not None and str(c).strip() and str(c).strip().lower() != "nan")
        ]

        # 1) remove unknown option(s)
        non_unknown = [c for c in choices if not _is_unknown_choice(c)]

        # 2) prefer entity choices that appear in context (if possible)
        hits = [(c, _first_match_pos(context, c)) for c in non_unknown]
        hits_in_ctx = [c for (c, pos) in hits if pos >= 0]

        if len(hits_in_ctx) >= 2:
            entity_choices = hits_in_ctx[:2]
        elif len(non_unknown) >= 2:
            entity_choices = non_unknown[:2]
        else:
            entity_choices = choices[:2]  # last resort

        target_loc = row.get("target_loc", None)
        target_text = ""

        # resolve target_text
        t = _to_intish(target_loc)
        if t is not None:
            # common: 0/1 means index among the TWO entities
            if t in (0, 1) and len(entity_choices) >= 2:
                target_text = entity_choices[t]
            # else: maybe it indexes ans0/ans1/ans2 directly
            elif 0 <= t < len(choices):
                target_text = choices[t]
        elif isinstance(target_loc, str) and target_loc.strip().upper() in ("A", "B", "C"):
            idx = {"A": 0, "B": 1, "C": 2}[target_loc.strip().upper()]
            if 0 <= idx < len(choices):
                target_text = choices[idx]
        elif isinstance(target_loc, str) and target_loc.strip():
            tl = _norm(target_loc)
            for c in entity_choices:
                if _norm(c) == tl:
                    target_text = c
                    break
            if not target_text:
                target_text = target_loc

        if _is_unknown_choice(target_text):
            target_text = ""

        # resolve other_text (must be the OTHER of the two entities)
        other_text = ""
        if len(entity_choices) >= 2 and target_text:
            tnorm = _norm(target_text)
            others = [c for c in entity_choices if _norm(c) != tnorm]
            other_text = others[0] if others else (entity_choices[1] if _norm(entity_choices[0]) == tnorm else entity_choices[0])
        else:
            other_text = entity_choices[1] if len(entity_choices) > 1 else ""

        return str(target_text), str(other_text)



 
    target_texts, other_texts, t_pos, o_pos, t_order = [], [], [], [], []

    for _, row in out.iterrows():
        context = row.get("context", "")
        target_text, other_text = _row_target_other(row)

        tp = _first_match_pos(context, target_text)
        op = _first_match_pos(context, other_text)

        if tp >= 0 and op >= 0:
            order = "first" if tp < op else "second"
        else:
            order = "unknown"

        target_texts.append(target_text)
        other_texts.append(other_text)
        t_pos.append(tp)
        o_pos.append(op)
        t_order.append(order)

    out["target_entity_text"] = target_texts
    out["other_entity_text"] = other_texts
    out["target_char_start"] = t_pos
    out["other_char_start"] = o_pos
    out["target_position"] = t_order
    return out

def prepare_df_from_hf(
    dataset_id: str | None,
    metadata_csv_path: str,
    revision: str | None = None,
) -> pd.DataFrame:
    raw = load_bbq_from_hf(dataset_id=dataset_id, revision=revision)
    md = load_metadata(metadata_csv_path)

    # Normalize join keys
    for col in ["category", "example_id", "question_index"]:
        if col not in raw.columns:
            raise KeyError(f"HF dataset missing '{col}'")
        raw[col] = raw[col].astype(str).str.strip()
        if col in md.columns:
            md[col] = md[col].astype(str).str.strip()
        else:
            raise KeyError(f"metadata CSV missing '{col}'")

    # Drop duplicate rows in HF data
    raw = raw.drop_duplicates(subset=["category", "question_index", "example_id"])

    # Deduplicate metadata; keep first non-NA target_loc per triple
    dup_md = (
        md.groupby(["category", "question_index", "example_id"])
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    if not dup_md.empty:
        md = (
            md.sort_values(["category", "question_index", "example_id"])
            .groupby(["category", "question_index", "example_id"], as_index=False)
            .agg(
                target_loc=(
                    "target_loc",
                    lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA,
                )
            )
        )
    else:
        md = md[["category", "question_index", "example_id", "target_loc"]].copy()

    # Merge HF data with metadata
    df = raw.merge(
        md,
        on=["category", "question_index", "example_id"],
        how="left",
        validate="m:1",
    )
    golds_idx, scs_idx, sis_idx, cc3, tgt_cond = [], [], [], [], []

    # If you later also gate by HF context_condition, you can precompute hf_bin here.
    # For now we keep your existing pure-index logic.

    for r in df.to_dict(orient="records"):
        choices = choices_from_ans_fields(r)
        n = len(choices)

        # 1) Unknown index: ONLY via text, no hard-coded default
        unk_idx = next((j for j in range(n) if _is_unknown_choice(choices[j])), None)

        # 2) Stereotype-consistent index from metadata
        tl = r.get("target_loc", pd.NA)
        sc_idx = int(tl) if pd.notna(tl) and 0 <= int(tl) < n else None

        # 3) Anti-stereotype index = first index that is not stereo and not unknown
        si_candidates = [j for j in range(n) if j != sc_idx and j != unk_idx]
        si_idx = si_candidates[0] if si_candidates else None

        # 4) Gold (correct) index from numeric label
        lab = r.get("label", None)
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            gold_idx = None
        else:
            gold_idx = int(lab) if 0 <= int(lab) < n else None

        # 5) Determine 3-way context condition using indices
        if gold_idx is not None and unk_idx is not None and gold_idx == unk_idx:
            cc = "AMBIG"
        elif gold_idx is not None and sc_idx is not None and gold_idx == sc_idx:
            cc = "DISAMBIG_STEREO"
        elif gold_idx is not None and si_idx is not None and gold_idx == si_idx:
            cc = "DISAMBIG_ANTI"
        else:
            cc = "DISAMBIG"

        # 6) Target condition label (relative to stereotype)
        if gold_idx is None:
            tc = ""
        elif sc_idx is not None and gold_idx == sc_idx:
            tc = "SC"
        elif si_idx is not None and gold_idx == si_idx:
            tc = "SI"
        elif unk_idx is not None and gold_idx == unk_idx:
            tc = "UNK"
        else:
            tc = ""

        golds_idx.append(gold_idx)
        scs_idx.append(sc_idx)
        sis_idx.append(si_idx)
        cc3.append(cc)
        tgt_cond.append(tc)

    # golds_idx, scs_idx, sis_idx, cc3 = [], [], [], []

    # for r in df.to_dict(orient="records"):
    #     choices = choices_from_ans_fields(r)
    #     n = len(choices)

    #     # 1) Unknown index: ONLY via text, no hard-coded default
    #     unk_idx = next((j for j in range(n) if _is_unknown_choice(choices[j])), None)

    #     # 2) Stereotype-consistent index from metadata
    #     tl = r.get("target_loc", pd.NA)
    #     sc_idx = int(tl) if pd.notna(tl) and 0 <= int(tl) < n else None

    #     # 3) Anti-stereotype index = first index that is not stereo and not unknown
    #     si_candidates = [j for j in range(n) if j != sc_idx and j != unk_idx]
    #     si_idx = si_candidates[0] if si_candidates else None

    #     # 4) Gold (correct) index from numeric label
    #     lab = r.get("label", None)
    #     if lab is None or (isinstance(lab, float) and np.isnan(lab)):
    #         gold_idx = None
    #     else:
    #         gold_idx = int(lab) if 0 <= int(lab) < n else None

    #     # 5) Determine 3-way context condition using indices
    #     if gold_idx is not None and unk_idx is not None and gold_idx == unk_idx:
    #         cc = "AMBIG"
    #     elif gold_idx is not None and sc_idx is not None and gold_idx == sc_idx:
    #         cc = "DISAMBIG_STEREO"
    #     elif gold_idx is not None and si_idx is not None and gold_idx == si_idx:
    #         cc = "DISAMBIG_ANTI"
    #     else:
    #         cc = "DISAMBIG"

    #     golds_idx.append(gold_idx)
    #     scs_idx.append(sc_idx)
    #     sis_idx.append(si_idx)
    #     cc3.append(cc)

    # # Optionally keep the index columns for debugging
    # df["gold_idx"] = golds_idx
    # df["sc_idx"] = scs_idx
    # df["si_idx"] = sis_idx

    # # Map indices back to letters for compatibility with older code
    # df["gold_label"] = [
    #     VALID_LETTERS[i] if i is not None and 0 <= i < len(VALID_LETTERS) else ""
    #     for i in golds_idx
    # ]
    # df["sc_label"] = [
    #     VALID_LETTERS[i] if i is not None and 0 <= i < len(VALID_LETTERS) else None
    #     for i in scs_idx
    # ]
    # df["si_label"] = [
    #     VALID_LETTERS[i] if i is not None and 0 <= i < len(VALID_LETTERS) else None
    #     for i in sis_idx
    # ]
    # df["context_condition_3"] = cc3

    df["gold_idx"] = golds_idx
    df["sc_idx"]   = scs_idx
    df["si_idx"]   = sis_idx

    df["gold_label"] = [
        VALID_LETTERS[i] if i is not None and 0 <= i < len(VALID_LETTERS) else ""
        for i in golds_idx
    ]
    df["sc_label"] = [
        VALID_LETTERS[i] if i is not None and 0 <= i < len(VALID_LETTERS) else None
        for i in scs_idx
    ]
    df["si_label"] = [
        VALID_LETTERS[i] if i is not None and 0 <= i < len(VALID_LETTERS) else None
        for i in sis_idx
    ]
    df["context_condition_3"] = cc3
    df["target_condition"] = tgt_cond   # <--- NEW

    # Normalize some categorical fields
    for col in ["category", "question_polarity"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()

    return df



# ----------------- EMBEDDER -----------------
def _should_skip_sentence_transformers(name: str) -> bool:
    # This function checks if a given model name should not use sentence-transformers.
    name_l = name.lower()
    bad_markers = [
        "google/embeddinggemma",
        "google/embedding-gemma",
        "google/embedding-gecko",
        "gecko-embedding",
    ]
    return any(m in name_l for m in bad_markers)


try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


class Embedder:
    """
    Wrapper around HF / sentence-transformers to get a consistent .encode()
    provides a consistent interface for encoding text into embeddings. 
    It handles several configurations and options like device and data type.
    """
    def __init__(self, name_or_path: str, device: str = "auto", dtype: str = "auto", batch_size: int = 64):
        from transformers import AutoTokenizer, AutoModel

        self.name = name_or_path
        self.bs = batch_size
        name_l = self.name.lower()
        self.is_gemma = "google/embeddinggemma" in name_l or "google/embeddinggemma-300m" in name_l

# devies
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if dtype == "auto":
            if self.is_gemma:
                self.dtype = torch.float32
            else:
                if self.device == "cuda":
                    self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                else:
                    self.dtype = torch.float32
        else:
            self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

        self.is_sentence_transformer = False
        self.st_model = None

# ST v HF
        if self.is_gemma:
            if not _HAS_ST:
                raise RuntimeError("google/embeddinggemma-300m needs sentence-transformers.")
            self.st_model = SentenceTransformer(self.name, device=self.device)
            self.is_sentence_transformer = True
            self.tok = None
            self.model = None
        else:
            if _HAS_ST and not _should_skip_sentence_transformers(self.name):
                try:
                    self.st_model = SentenceTransformer(self.name, device=self.device)
                    self.is_sentence_transformer = True
                except Exception:
                    self.st_model = None
                    self.is_sentence_transformer = False

            if not self.is_sentence_transformer:
                self.tok = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True, use_fast=True)
                if "qwen3-embedding" in name_l or "qwen/" in name_l:
                    try:
                        self.tok.padding_side = "left"
                    except Exception:
                        pass
                self.model = AutoModel.from_pretrained(
                    self.name,
                    torch_dtype=self.dtype if self.device != "mps" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                self.model = self.model.to(self.device)
                self.model.eval()
            else:
                self.tok = None
                self.model = None

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = None, normalize: bool = True) -> np.ndarray:
    # This function takes a list of texts and returns their embeddings, either using sentence-transformers or a Hugging Face model. 
    # It includes options for normalization and batching.

        bs = batch_size or self.bs
        if self.is_sentence_transformer: # The method self.st_model.encode uses the sentence-transformers library's encoding method.
            embs = self.st_model.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=normalize, # The normalize_embeddings argument is directly passed to this method, which will normalize the embeddings if normalize is True.
                # This ensures that the embeddings have unit length (norm of 1).
                show_progress_bar=False,
            )
            return embs
        else: # If the model is not from sentence-transformers, it relies on Hugging Face's Transformers library:
            all_vecs = []
            for i in range(0, len(texts), bs):
                chunk = texts[i: i + bs]
                tokens = self.tok( # texts are tokenized using self.tok.
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**tokens)
                last_hidden = outputs.last_hidden_state # The tokens are passed through the model to get last_hidden_state, which holds the embeddings.
                mask = tokens.attention_mask.unsqueeze(-1) # Masks are used to sum the hidden states for each token, and the result is mean-pooled to obtain fixed-length embeddings for each text.
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                mean_pooled = (summed / counts).detach().to("cpu").numpy()
                all_vecs.append(mean_pooled) # Embeddings are stored in all_vecs and then stacked vertically using np.vstack.
            embs = np.vstack(all_vecs)
            if normalize: # , normalization is manually performed after stacking the embeddings, delegated to the encode method of the sentence-transformers library
                norms = np.linalg.norm(embs, axis=1, keepdims=True) #  computes the norm (length) of each vector.
                norms[norms == 0.0] = 1.0 # Norms are adjusted to prevent division by zero.
                embs = embs / norms # Embeddings are normalized by dividing each vector by its norm, ensuring that each embedding has a unit length.
            return embs

    def encode_queries(self, texts: List[str], batch_size: int = None, normalize: bool = True) -> np.ndarray:
        # fall back to encode
        return self.encode(texts, batch_size=batch_size, normalize=normalize)

    def encode_docs(self, texts: List[str], batch_size: int = None, normalize: bool = True) -> np.ndarray:
        # fall back to encode
        return self.encode(texts, batch_size=batch_size, normalize=normalize)


def _detect_condition_col(df: pd.DataFrame) -> str:
# This function tries to find a column in a DataFrame that holds condition/context information.
    for c in ["context_condition_3", "context_condition", "condition", "cc"]:
        if c in df.columns:
            return c
    raise KeyError("No context condition column found")


def load_df_any(path: str) -> pd.DataFrame:
# This utility function loads a DataFrame from either a CSV or parquet file, depending on the file extension.
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
