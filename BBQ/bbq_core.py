# bbq_core.py


import os
import math
import json
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


def is_unknown_text(s: str) -> bool:
    # Function to check if a string indicates unknown information
    if not isinstance(s, str):
        return False
    t = s.strip().lower()
    markers = [
        "unknown",
        "undetermined",
        "cannot be determined",
        "can not be determined",
        "can't be determined",
        "cannot determine",
        "can not determine",
        "can't determine",
        "not determined",
        "can't tell",
        "cannot tell",
        "not known",
        "insufficient information",
        "not enough information",
        "cannot be known",
        "cannot know",
        "indeterminate",
        "undecidable",
        "unanswerable",
    ]
    return any(m in t for m in markers)


def choices_from_ans_fields(row: Dict[str, Any]) -> List[str]:
#This function extracts answer fields from a dictionary row and returns them as a list of strings.
    out = []
    for k in ["ans0", "ans1", "ans2"]:
        if k in row and isinstance(row[k], str) and row[k].strip():
            out.append(row[k].strip())
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


def prepare_df_from_hf(
    # main data preparation function
    # combines dataset loading and metadata loading, then processes and merges them into a single DataFrame. 
    # It also generates new columns like 'gold_label', 'sc_label', and 'si_label' based on certain conditions
    dataset_id: str | None,
    metadata_csv_path: str,
    revision: str | None = None,
    default_unknown_index: int = 2,
) -> pd.DataFrame:
    raw = load_bbq_from_hf(dataset_id=dataset_id, revision=revision)
    md = load_metadata(metadata_csv_path)

    for col in ["category", "example_id", "question_index"]:
        if col not in raw.columns:
            raise KeyError(f"HF dataset missing '{col}'")
        raw[col] = raw[col].astype(str).str.strip()
        if col in md.columns:
            md[col] = md[col].astype(str).str.strip()
        else:
            raise KeyError(f"metadata CSV missing '{col}'")

    raw = raw.drop_duplicates(subset=["category", "question_index", "example_id"])

    dup_md = (
        md.groupby(["category", "question_index", "example_id"])
        .size()
        .reset_index(name="n")
        .query("n>1")
    )
    if not dup_md.empty:
        md = (
            md.sort_values(["category", "question_index", "example_id"])
            .groupby(["category", "question_index", "example_id"], as_index=False)
            .agg(target_loc=("target_loc", lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA))
        )
    else:
        md = md[["category", "question_index", "example_id", "target_loc"]].copy()

    df = raw.merge(md, on=["category", "question_index", "example_id"], how="left", validate="m:1")

    golds, scs, sis, cc3 = [], [], [], []
    for r in df.to_dict(orient="records"):
        choices = choices_from_ans_fields(r)
        n = len(choices)

        unk_idx = next((j for j in range(n) if is_unknown_text(choices[j])), None)
        if unk_idx is None and n == 3 and 0 <= default_unknown_index < 3:
            unk_idx = default_unknown_index

        tl = r.get("target_loc", pd.NA)
        sc_idx = int(tl) if pd.notna(tl) and 0 <= int(tl) < n else None

        si_idx = next(
            (
                j
                for j in range(n)
                if (sc_idx is None or j != sc_idx) and (unk_idx is None or j != unk_idx)
            ),
            None,
        )

        sc_letter = VALID_LETTERS[sc_idx] if sc_idx is not None else None
        si_letter = VALID_LETTERS[si_idx] if si_idx is not None else None
        gold_letter = gold_letter_from_numeric_label(r, n)

        if gold_letter and (unk_idx is not None) and gold_letter == VALID_LETTERS[unk_idx]:
            cc = "AMBIG"
        elif gold_letter and (sc_letter is not None) and gold_letter == sc_letter:
            cc = "DISAMBIG_STEREO"
        elif gold_letter and (si_letter is not None) and gold_letter == si_letter:
            cc = "DISAMBIG_ANTI"
        else:
            cc = "DISAMBIG"

        golds.append(gold_letter)
        scs.append(sc_letter)
        sis.append(si_letter)
        cc3.append(cc)

    df["gold_label"] = golds
    df["sc_label"] = scs
    df["si_label"] = sis
    df["context_condition_3"] = cc3

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
