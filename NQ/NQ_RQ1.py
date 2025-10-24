# NQ_RQ1.py
from datasets import load_dataset, concatenate_datasets
import os, re, json, argparse, torch, time, random, requests
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util as st_util
from scipy import stats  # t-tests

from openai import APIError, APITimeoutError, RateLimitError
try:
    from openai import NotFoundError
except Exception:
    class NotFoundError(APIError): pass

SEED = 42
np.random.seed(SEED)

# -------- simple phase logger (timestamps) ----------
def _phase(msg: str, _t=[None]):
    import time
    now = time.time()
    if _t[0] is None:
        print(f"[PHASE] {msg}"); _t[0] = now
    else:
        print(f"[PHASE] {msg} (+{now - _t[0]:.2f}s)"); _t[0] = now


# ====================================================
#  dataset loading
# ====================================================
UNKNOWN_ANSWERS = {
    "can't be determined", "not enough info", "unknown",
    "can't answer", "cannot answer", "cannot be determined",
    "not known", "undetermined"
}

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
    csv = os.path.join(out_dir, "granite.csv")
    pq  = os.path.join(out_dir, "granite.parquet")

    _arr_pat = re.compile(r"array\(\s*(\[[^\]]*\])\s*,\s*dtype=object\s*\)")
    def _strip_numpy_arrays(s: str) -> str:
        return _arr_pat.sub(r"\1", s)

    def _string_to_jsonable(s: str) -> str:
        # converts string to json
        s = (s or "").strip()
        if not s or (s[0] not in "{[" or s[-1] not in "}]"):
            return ""
        s = s.replace('""', '"')
        s = _strip_numpy_arrays(s)
        s = s.replace("'", '"')
        return s

    def _parse_obj_cell(x):
        if isinstance(x, dict): return x
        if not isinstance(x, str): return {}
        js = _string_to_jsonable(x)
        if not js: return {}
        try:
            return json.loads(js)
        except Exception:
            return {}

    def _coerce_cols(df):
        # populate answer_info and additional_metadata columns 
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

# ====================================================
# embedding model data input
# ====================================================
def _truncate_texts(texts: List[str], max_chars: int) -> List[str]:
    # character cap for embedding model input
    if max_chars is None or max_chars <= 0:
        return texts
    return [(t if t is not None else "")[:max_chars] for t in texts]


def load_embed_model(model_name: str, device: str = "cuda") -> SentenceTransformer:
    # Wraps SentenceTransformer load and prints device. Purpose: local (GPU/CPU) embedding fallback when not using RITS.
    print(f"[MODEL] Loading SentenceTransformer on {device}: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    try:
        print("[DEBUG] model device:", next(model.auto_model.parameters()).device)
    except Exception:
        pass
    return model

# ====================================================
# rits
# ====================================================
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

class RITSClient:
    """
    RITS model discovery with optional endpoint override and safe retries.
    If discovery fails (5xx), we can still proceed when an endpoint override is given.
    """
    def __init__(self, api_key: str, default_timeout: float = 120.0,
                 endpoint_override: Optional[str] = None):
        if not api_key:
            raise ValueError("RITS API key missing (pass --rits-key or set RITS_API_KEY)")
        self.api_key = api_key
        self._models = None
        self.default_timeout = float(default_timeout)
        # self._override = endpoint_override or {}
        self.endpoint_override = endpoint_override

    def _fetch_models(self, tries: int = 3, base: float = 1.5) -> Optional[Dict[str, str]]:
        # calls rits discovery endpoint, find the right base URL for a given RITS model.
        url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
        last_err = None
        for i in range(tries):
            try:
                resp = requests.get(url, headers={"RITS_API_KEY": self.api_key}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                return {m["model_name"]: m["endpoint"] for m in data}
            except Exception as e:
                last_err = e
                sleep = (base ** i) + random.uniform(0.0, 0.4)
                print(f"[RITS] discovery failed (try {i+1}/{tries}): {type(e).__name__} → retrying in {sleep:.1f}s")
                time.sleep(sleep)
        print(f"[RITS] discovery ultimately failed → {last_err}")
        return None

    @property
    def models(self) -> dict:
        # Lazy-loads and caches discovery results.
        if self._models is None:
            m = self._fetch_models()
            # If discovery failed, keep empty dict; endpoint overrides may still allow run
            self._models = m or {}
        return self._models

    def client_for(self, model_name: str, timeout: float):
        # Constructs an OpenAI-style client pointed at the resolved endpoint
        if _OpenAI is None:
            raise RuntimeError("openai>=1.0 is required for RITS. pip install openai")
        if self.endpoint_override:
            base_url = self.endpoint_override.rstrip("/") + "/v1"
        else:
            if model_name not in self.models:
                avail = sorted(self.models.keys())[:12]
                raise KeyError(f"Unknown RITS model: '{model_name}'. Available (sample): {avail} ...")
            base_url = f"{self.models[model_name]}/v1"
        return _OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            default_headers={"RITS_API_KEY": self.api_key},
            timeout=float(timeout),
        )


def _retry(func, *, tries: int, base: float = 1.5, tag: str = "RITS"):
    # Generic retry wrapper for transient OpenAI/RITS API errors.
    for i in range(tries):
        try:
            return func()
        except (APITimeoutError, RateLimitError, APIError, NotFoundError) as e:
            if i == tries - 1:
                raise
            sleep = (base ** i) + random.uniform(0.0, 0.4)
            print(f"[{tag}] transient error ({type(e).__name__}): retry {i+1}/{tries} in {sleep:.1f}s")
            time.sleep(sleep)

class RITSEmbedder:
    def __init__(self, rits: RITSClient, model_name: str,
                 # Stores client/config for embedding via RITS. Purpose: make a SentenceTransformer-like adapter.
                 timeout: float = 300.0, max_retries: int = 6,
                 min_batch: int = 4, inter_batch_sleep: float = 0.15,
                 max_chars: Optional[int] = None):
        self.rits = rits
        self.model_name = model_name
        self.client = rits.client_for(model_name, timeout=timeout)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.min_batch = int(min_batch)
        self.sleep = float(inter_batch_sleep)
        self.max_chars = max_chars

    def _embed_once(self, chunk: List[str]) -> np.ndarray:
        #Single API call to get embeddings for a small batch; returns a stacked float32 numpy array.
        if self.max_chars is not None:
            chunk = [truncate_for_embed(x, self.max_chars) for x in chunk]
        resp = _retry(lambda: self.client.embeddings.create(model=self.model_name, input=chunk),
                      tries=self.max_retries, tag="RITS/emb")
        return np.vstack([np.asarray(d.embedding, dtype=np.float32) for d in resp.data])

    def _embed_chunk_adaptive(self, chunk: List[str]) -> np.ndarray:
        # when hitting timelimit/timeout recursively halves the batch until it fits; then re-stitches. instead of failing.
        size = len(chunk)
        while True:
            try:
                return self._embed_once(chunk)
            except APITimeoutError:
                if size <= self.min_batch:
                    if size == 1:
                        return self._embed_once(chunk)
                    mid = size // 2
                    left = self._embed_chunk_adaptive(chunk[:mid])
                    right = self._embed_chunk_adaptive(chunk[mid:])
                    return np.vstack([left, right])
                mid = size // 2
                print(f"[RITS] timeout on batch={size}; splitting → {mid}+{size-mid}")
                left = self._embed_chunk_adaptive(chunk[:mid])
                right = self._embed_chunk_adaptive(chunk[mid:])
                return np.vstack([left, right])

    def encode(self, texts: List[str], batch_size: int = 64,
               convert_to_numpy: bool = True, show_progress_bar: bool = True, **_):
        # Batches through texts, calls _embed_chunk_adaptive, optional sleep between chunks; returns numpy array. 
        # drop-in replacement for SentenceTransformer.encode
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rng = range(0, len(texts), batch_size)
        if show_progress_bar:
            from tqdm.auto import tqdm as _tqdm
            rng = _tqdm(rng, total=(len(texts)+batch_size-1)//batch_size, desc="Batches") # buildng All unique questions (embedded once each). All answers (original + injections).
            #  each “Batches” bar is one call to batch_embed_any (first for questions, then for answers).
        outs = []
        for i in rng:
            chunk = texts[i:i+batch_size]
            arr = self._embed_chunk_adaptive(chunk)
            outs.append(arr)
            if self.sleep > 0:
                time.sleep(self.sleep)
        out = np.vstack(outs)
        return out if convert_to_numpy else out.tolist()

def load_embedder_from_args(args):
    # Chooses RITS or local sentence-transformer backend from CLI flags; returns an object with .encode() 
    # Purpose: single entry to configure embeddings.
    if getattr(args, "use_rits_embed", False):
        print(f"[MODEL] Using RITS embeddings: {args.rits_embed_model}")
        try:
            rits = RITSClient(api_key=args.rits_key, default_timeout=args.rits_timeout,
                              endpoint_override=args.rits_endpoint)
            # only touch discovery if no override provided
            if not args.rits_endpoint:
                _ = rits.models  # validate discovery
            return RITSEmbedder(
                rits, args.rits_embed_model,
                timeout=args.rits_timeout,
                max_retries=args.rits_max_retries,
                min_batch=args.rits_min_batch,
                inter_batch_sleep=args.rits_sleep
            )
        except Exception as e:
            print(f"[RITS] Failed to initialize RITS ({e}); falling back to local SentenceTransformer.")
            # fall back to local
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Loading SentenceTransformer on {device}: {args.model}")
    model = SentenceTransformer(args.model, device=device)
    try:
        print("[DEBUG] model device:", next(model.auto_model.parameters()).device)
    except Exception:
        pass
    return model



# ====================================================
# Truncation helpers (avoid 512-token errors)
# ====================================================
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def truncate_for_embed(text: str, max_chars: int) -> str:
    """Sentence-aware truncation with hard cap. Ensure that chopping long answers is still natural."""
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    # try sentence-wise accumulation
    parts = _SENT_SPLIT.split(s)
    out, total = [], 0
    for p in parts:
        p2 = p.strip()
        if not p2:
            continue
        if total + len(p2) + 1 > max_chars:
            break
        out.append(p2)
        total += len(p2) + 1
    if not out:
        return s[:max_chars]
    return " ".join(out)[:max_chars]

# ====================================================
# Cosine similarity over BBQ 
# ====================================================
def get_cosine_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    # Encodes the two texts and returns their cosine similarity
    embs = model.encode([text1 or "", text2 or ""], convert_to_tensor=True)
    return st_util.pytorch_cos_sim(embs[0], embs[1]).item()

def compute_bbq_cosines(bbq_df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    # Vectorizes (context ↔ question) over full BBQ, saving key columns. Purpose: build granite.* once for identity harvesting.
    req_cols = ["context","question"]
    missing = [c for c in req_cols if c not in bbq_df.columns]
    if missing:
        raise ValueError(f"BBQ is missing required columns: {missing}")
    records = []
    it = tqdm(bbq_df.itertuples(index=False), total=len(bbq_df), desc="Cosine BBQ (context ↔ question)")
    for row in it:
        ex = row._asdict()
        sim = get_cosine_similarity(str(ex.get("context","")), str(ex.get("question","")), model)
        out = {
            "cosine_similarity": sim,
            "category": ex.get("category"),
            "example_id": ex.get("example_id"),
            "question_index": ex.get("question_index"),
            "context_condition": ex.get("context_condition"),
            "question_polarity": ex.get("question_polarity"),
            "context": ex.get("context"),
            "question": ex.get("question"),
            "ans0": ex.get("ans0"),
            "ans1": ex.get("ans1"),
            "ans2": ex.get("ans2"),
            "label": ex.get("label"),
            "answer_info": ex.get("answer_info"),
            "additional_metadata": ex.get("additional_metadata"),
        }
        records.append(out)
    granite = pd.DataFrame(records)
    print(f"[BBQ] Built cosine table: {len(granite)} rows, columns={len(granite.columns)}")
    return granite

# ====================================================
# helper functions for identifying/tagging identity terms 
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
    # Guarantees sentence-final punctuation
    s = (s or "").rstrip()
    return s if s.endswith((".", "!", "?")) else s + "."

def looks_like_name(entity: str) -> bool:
    # does the entity look like a name or not 
    if not entity or not isinstance(entity, str): return False
    toks = entity.strip().split()
    if len(toks) < 2: return False # not a name, since names often have 2 parts, 
    role_heads = {"man","woman","boy","girl","person","people"}
    if any(t.lower() in role_heads for t in toks): return False # not a name if it includes a role head 
    uppers = sum(t[:1].isupper() for t in toks if t)
    return uppers >= max(2, int(0.7 * len(toks))) # if two tokens have a capital letter, it is likely a name 

def is_full_np(s: str) -> bool: 
    if not s: return False # if string looks like a noun phrase 
    e = s.strip().lower()
    return e.startswith(("a ","an ","the ","this ","that ","these ","those ",
                         "person ","person-","man","woman","boy","girl","teen",
                         "elderly","older","younger","middle-aged","pregnant",
                         "disabled","deaf","blind","wheelchair"))

def normalize_ses(lab: str) -> str:
    # SES category involves unstandardized references to class, this function fixes it
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

def realize_identity_ref(entity: str, label: str, category: str) -> Tuple[str, bool]:
    # Returns (phrase, is_name_flag) turning BBQ entities/labels into fluent NPs
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
        if low in {"autism","autism spectrum disorder","ptsd","bpd","schizophrenia","anxiety","depression","cerebral palsy"}:
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

# ========= templates for identity injection, and random injection =========
INJ_TEMPLATES_BEGIN = {"acc_to": "According to {PHRASE}, "}
INJ_TEMPLATES_END   = {"acc_to": " According to {PHRASE}."}

RAND_ADJS  = ["ordinary", "generic", "blue", "plain", "simple", "typical", "random", "pink", "tinted", "neutral"]
RAND_NOUNS = ["object", "item", "widget", "thing", "note", "statement", "remark", "comment", "train", "bottle", "slack", "bag"]

def make_random_phrase(rng: np.random.Generator) -> str:
    # picks one adjective and one noun at random from two pre‑defined lists
    a = rng.choice(RAND_ADJS)
    n = rng.choice(RAND_NOUNS)
    return f"{choose_article(a)} {a} {n}"

def inject_with_template(answer: str, phrase: str, where: str, template_name: str) -> str:
    ans = (answer or "").strip()
    if where == "begin":
        tpl = INJ_TEMPLATES_BEGIN.get(template_name, INJ_TEMPLATES_BEGIN["acc_to"])
        if ans: ans = ans[0].upper() + ans[1:]
        return tpl.replace("{PHRASE}", phrase) + ans
    else:
        tpl = INJ_TEMPLATES_END.get(template_name, INJ_TEMPLATES_END["acc_to"])
        return ensure_terminal(ans) + tpl.replace("{PHRASE}", phrase)

# ====================================================
# Embedding utils (with truncation)
# ====================================================
def l2_normalize(arr: np.ndarray) -> np.ndarray:
    # Row-normalizes vectors to unit length 
    arr = np.asarray(arr)
    if arr.ndim == 1:
        denom = np.linalg.norm(arr) + 1e-12
        return arr / denom
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / denom

def batch_embed_any(texts, model, batch_size=64, normalize=True, progress=True, desc="Embedding", max_chars=None):
    # Uniform embedding wrapper for either backend; applies char truncation; returns normalized embeddings. Purpose: single path the rest of the code calls.
    texts = _truncate_texts(texts, max_chars or 0)
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=progress) # calls encoder
    return l2_normalize(vecs) if normalize else vecs


# ====================================================
# identifies questions and answers in datasets
# ====================================================
_Q_KEYS = ["query", "question", "q", "title"]
_A_KEYS = ["answer", "long_answer", "passage", "context", "document", "text", "content", "body"]

def get_question(ex: dict) -> str:
    # finds question regardless of dataset format
    for k in _Q_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for _, v in ex.items():
        if isinstance(v, dict):
            for kk in _Q_KEYS:
                t = v.get(kk)
                if isinstance(t, str) and t.strip():
                    return t.strip()
    return ""

def choose_answer_sentence(ex: dict, max_chars: int = 1500) -> str:
    # Finds an answer/long answer, picks the first sentence (optionally second), and caps length
    text = ""
    for k in _A_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            text = v.strip(); break
        if isinstance(v, (list, tuple)) and v:
            joined = " ".join([str(x) for x in v if isinstance(x, str)]).strip()
            if joined: text = joined; break
        if isinstance(v, dict):
            for kk in ["text","content","span_text","body"]:
                t = v.get(kk)
                if isinstance(t, str) and v.get(kk, "").strip():
                    text = t.strip(); break
            if text: break
    if not text:
        for _, v in ex.items():
            if isinstance(v, dict):
                for kk in _A_KEYS + ["text","content","span_text","body"]:
                    t = v.get(kk)
                    if isinstance(t, str) and v.get(kk, "").strip():
                        text = t.strip(); break
                if text: break
    if not text:
        return ""
    sents = _SENT_SPLIT.split(text)
    if not sents:
        return text[:max_chars]
    s1 = (sents[0] or "").strip()
    if not s1:
        return ""
    if len(s1) < max_chars and len(sents) > 1:
        s2 = (sents[1] or "").strip()
        if s2:
            return (s1 + " " + s2)[:max_chars]
    return s1[:max_chars]

# ====================================================
# stats
# One-sample t-test → test if injection effect differs from zero.
#Cohen’s d → quantify how big that effect is.
#Not ANOVA → because not testing group differences, but each group vs baseline.
#Not just p-values → because effect size matters more than mere significance in huge datasets.
# ====================================================
# ANOVA
#  “Does the mean Δ differ depending on the identity_category?”
# Which social dimensions (religion, race, SES, gender, etc.) show the strongest or most consistent bias overall

# Only keep injected identity rows (not random, not originals)
 

def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    # Cohen’s d says how large is the effect in standard-deviation units.
    d = x - y
    if d.size < 2: return np.nan
    sd = d.std(ddof=1)
    return d.mean() / sd if sd > 0 else np.nan

def _ci95_from_std(std, n):
    # Computes 95% CI half-width from std & n to plot error bars consistently.
    # expects already-aggregated std & n.
    std = np.asarray(std, float)
    n = np.asarray(n, float)
    se = std / np.sqrt(np.clip(n, 1, None))
    return 1.96 * se

def harvest_bbq_pairs_from_granite(
        # build a diverse identity set to inject
        # each pair is a tuple of (entity, label, category)
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
    if granite_df is None or granite_df.empty:
        print("[RQ1] Granite empty → no identity pairs harvested.")
        return []

    it = tqdm(granite_df.itertuples(index=False), total=len(granite_df),
              desc="Harvesting identities", disable=not progress)

    raw_count = 0
    for row in it:
        row = row._asdict()
        cat = row.get("category", "unknown")

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
        if isinstance(groups, np.ndarray): groups = groups.tolist()
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
        pairs_by_cat[cat] = uniq

    total_uniq = sum(len(v) for v in pairs_by_cat.values())
    print(f"[RQ1] harvested raw pairs: {raw_count} across {len(pairs_by_cat)} categories")
    print(f"[RQ1] harvested uniq pairs: {total_uniq} across {len(pairs_by_cat)} categories")

    if per_category is not None:
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

# =========================
# Plot 
# =========================
def plot_rq1_identity_vs_random(norm_path: str, out_path: str):
    # From rq1_norm_identity_vs_random.csv, plots category means for Identity, Random, and (Identity−Random) with 95% CIs, split by begin/end. 
    # show identity effect vs template baseline, by category.
    df = pd.read_csv(norm_path)
    need = {"where","identity_category","delta_identity","delta_random","delta_norm"}
    missing = need - set(df.columns)
    if missing:
        print(f"[PLOT:RQ1:IVR] missing required columns in {norm_path}: {missing}")
        return
    def _ci(x):
        # works directly from raw data, but maybe REFACTOR
        x = np.asarray(x, float)
        if x.size < 2:
            return np.nan
        se = x.std(ddof=1) / np.sqrt(x.size)
        return 1.96 * se
    gb = (df.groupby(["identity_category","where"])
            .agg(mean_id=("delta_identity","mean"),
                 mean_rd=("delta_random","mean"),
                 mean_nm=("delta_norm","mean"),
                 n=("delta_norm","count"),
                 ci_id=("delta_identity", _ci),
                 ci_rd=("delta_random",  _ci),
                 ci_nm=("delta_norm",   _ci))
            .reset_index())
    cats = sorted(gb["identity_category"].unique())
    where_levels = ["begin","end"]
    panels = {}
    for w in where_levels:
        sub = gb[gb["where"] == w].set_index("identity_category").reindex(cats)
        panels[w] = sub
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, constrained_layout=True)
    labels = ["Identity", "Random", "Identity − Random"]
    keys   = ["mean_id", "mean_rd", "mean_nm"]
    errs   = ["ci_id",   "ci_rd",   "ci_nm"  ]
    for ax, w in zip(axes, where_levels):
        sub = panels[w]
        x = np.arange(len(cats))
        width = 0.25
        for j, (lab, k, e) in enumerate(zip(labels, keys, errs)):
            vals = sub[k].values
            yerr = sub[e].values
            ax.bar(x + (j-1)*width, vals, width, label=lab, yerr=yerr, capsize=3)
        ax.axhline(0, ls="--", c="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_title(f"{w} injection")
        ax.set_ylabel("Δ cosine vs. original (mean ±95% CI)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.15), ncol=3)
    fig.suptitle("RQ1: Identity vs Random by Category and Position", y=1.04, fontsize=14)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT:RQ1:IVR] wrote → {out_path}")

def plot_rq1_by_category_mean(row_path: str, out_path: str):
    # From rq1_row_level.csv, computes category means of Δ vs original (begin/end) with 95% CIs. 
    # high-level view of injection impact by category.
    df = pd.read_csv(row_path)
    sub = df[df["answer_variant"].isin(["inject_begin","inject_end"])].copy()
    if sub.empty:
        print(f"[PLOT:RQ1] no injected rows found in {row_path}")
        return
    gb = (sub.groupby(["identity_category","answer_variant"], as_index=False)
              .agg(mean_delta=("delta_vs_original","mean"),
                   std=("delta_vs_original","std"),
                   n=("delta_vs_original","count")))
    gb["ci"] = _ci95_from_std(gb["std"], gb["n"])
    piv = gb.pivot(index="identity_category", columns="answer_variant", values="mean_delta")
    ci_piv = gb.pivot(index="identity_category", columns="answer_variant", values="ci")
    for col in ["inject_begin","inject_end"]:
        if col not in piv.columns: piv[col] = float("nan")
        if col not in ci_piv.columns: ci_piv[col] = float("nan")
    piv = piv[["inject_begin","inject_end"]].sort_values("inject_begin")
    ci_piv = ci_piv.loc[piv.index, ["inject_begin","inject_end"]]
    cats = list(piv.index); x = range(len(cats)); width = 0.42
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar([i - width/2 for i in x], piv["inject_begin"], width,
           yerr=ci_piv["inject_begin"], capsize=3, label="begin")
    ax.bar([i + width/2 for i in x], piv["inject_end"], width,
           yerr=ci_piv["inject_end"], capsize=3, label="end")
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_xticks(list(x)); ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Δ Cosine vs. Original")
    ax.set_title("RQ1: Identity Injection Effect by Category (mean ±95% CI)")
    ax.legend()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT:RQ1] wrote → {out_path}")

def plot_rq1_template_effects(norm_path: str, out_path: str):
    # plots different templates, if mroe are added, eventually
    df = pd.read_csv(norm_path)
    if df.empty:
        print(f"[PLOT:RQ1:TEMPL] nothing in {norm_path}")
        return
    g = (df.groupby(["where","template"], as_index=False)
           .agg(mean_norm=("delta_norm","mean"),
                std=("delta_norm","std"),
                n=("delta_norm","count")))
    g["ci"] = _ci95_from_std(g["std"], g["n"])
    templates = sorted(g["template"].unique())
    x = np.arange(len(templates))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12,5))
    g_b = g[g["where"]=="begin"].set_index("template").reindex(templates)
    g_e = g[g["where"]=="end"  ].set_index("template").reindex(templates)
    ax.bar(x - width/2, g_b["mean_norm"], width, yerr=g_b["ci"], capsize=3, label="begin")
    ax.bar(x + width/2, g_e["mean_norm"], width, yerr=g_e["ci"], capsize=3, label="end")
    ax.axhline(0, ls="--", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(templates, rotation=45, ha="right")
    ax.set_ylabel("Identity effect minus random (Δ_norm)")
    ax.set_title("RQ1: Template sensitivity (identity − random)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT:RQ1:TEMPL] wrote → {out_path}")

def plot_rq1_identity_vs_identity(ivs_path: str, out_path: str):
    # compare identities directly plots within-category differences between two identities’ Δ (A−B), begin vs end
    df = pd.read_csv(ivs_path)
    if df.empty:
        print(f"[PLOT:RQ1:IVI] nothing in {ivs_path}")
        return
    g = (df.groupby(["identity_category","where"], as_index=False)
           .agg(mean_diff=("delta_diff","mean"),
                std=("delta_diff","std"),
                n=("delta_diff","count")))
    g["ci"] = _ci95_from_std(g["std"], g["n"])
    piv = g.pivot(index="identity_category", columns="where", values="mean_diff").fillna(0.0)
    ci  = g.pivot(index="identity_category", columns="where", values="ci").reindex(piv.index)
    cats = list(piv.index); x = np.arange(len(cats)); width = 0.42
    fig, ax = plt.subplots(figsize=(12,6))
    if "inject_begin" in piv.columns:
        ax.bar(x - width/2, piv["inject_begin"], width, yerr=ci.get("inject_begin"), capsize=3, label="begin")
    if "inject_end" in piv.columns:
        ax.bar(x + width/2, piv["inject_end"], width, yerr=ci.get("inject_end"), capsize=3, label="end")
    ax.axhline(0, ls="--", c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Δ(identity A) − Δ(identity B)  (mean)")
    ax.set_title("RQ1: Identity vs Identity (within category)")
    ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close(fig)
    print(f"[PLOT:RQ1:IVI] wrote → {out_path}")

# ====================================================
# Core study
# ====================================================
def run_rq1_study1(
    nq_dataset,
    granite_df: pd.DataFrame,
    model: SentenceTransformer,
    sample_n_questions: int,
    per_category: int,
    begin_and_end: bool,
    qualitative_top_k: int,
    progress: bool,
    out_dir: str,
    embed_batch_size: int,
    pairs_per_question: int,
    max_row_level: int,
    embed_max_chars: int = 0,
    embed_max_tokens: int = 0,
    debug: bool = False,
    no_resume: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run RQ1: Identity injection drift study."""

    # --------------------------
    # Local helpers
    # --------------------------
    def _truncate_text(s: str) -> str:
        if not isinstance(s, str):
            s = "" if s is None else str(s)
        if embed_max_tokens and embed_max_tokens > 0:
            toks = re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)
            if len(toks) > embed_max_tokens:
                s = " ".join(toks[:embed_max_tokens])
        elif embed_max_chars and embed_max_chars > 0 and len(s) > embed_max_chars:
            s = s[:embed_max_chars]
        return s

    def _truncate_list(texts: List[str]) -> List[str]:
        return [_truncate_text(t) for t in texts]

    def pick_pairs_for_question(all_pairs, k, rng):
        if not all_pairs:
            return []
        k = min(k, len(all_pairs))
        idx = rng.choice(len(all_pairs), size=k, replace=False)
        return [all_pairs[i] for i in idx]

    # --------------------------
    # Harvest identity pairs
    # --------------------------
    pairs = harvest_bbq_pairs_from_granite(
        granite_df,
        max_pairs=None,
        per_category=per_category,
        shuffle_within_cat=True,
        seed=SEED,
        progress=progress,
    )
    cats = sorted({c for _, _, c in pairs}) if pairs else []
    print(f"[RQ1] harvested pairs: {len(pairs)} across {len(cats)} categories")

    # --------------------------
    # Collect usable NQ items
    # --------------------------
    ds = nq_dataset['train'] if isinstance(nq_dataset, dict) and 'train' in nq_dataset else nq_dataset
    try:
        total = len(ds)
    except Exception:
        ds = list(ds); total = len(ds)

    idxs = []
    scan_it = tqdm(range(total), total=total, desc="Scanning NQ for Q&A", disable=not progress)
    for i in scan_it:
        ex = ds[i]
        q = get_question(ex)
        a = choose_answer_sentence(ex)
        if q and a:
            idxs.append(i)
        if len(idxs) >= sample_n_questions:
            break
    print(f"[RQ1] usable NQ items: {len(idxs)}")
    if not idxs:
        print("[RQ1] No usable NQ items; skipping.")
        return {"row_level": pd.DataFrame()}

    # --------------------------
    # Pre-flight blow-up guard
    # --------------------------
    positions = ["begin", "end"] if begin_and_end else ["begin"]
    n_positions = len(positions)
    n_templates = max(len(INJ_TEMPLATES_BEGIN), len(INJ_TEMPLATES_END))
    base_pairs = max(0, pairs_per_question if pairs else 0)
    est_rows = len(idxs) * (1 + base_pairs * n_positions * n_templates * 2)
    print(f"[RQ1] pre-flight estimate rows ≈ {est_rows:,} (cap={max_row_level:,})")

    if est_rows > max_row_level and base_pairs > 0:
        denom = (len(idxs) * n_positions * n_templates * 2)
        k_max = int(max((max_row_level - len(idxs)) // denom, 0))
        if k_max < base_pairs:
            print(f"[RQ1] reducing pairs_per_question {base_pairs} → {k_max} to satisfy cap")
            pairs_per_question = k_max

    # --------------------------
    # Build / resume row-level table
    # --------------------------
    row_path = os.path.join(out_dir, "rq1_row_level.csv")
    need_cols = {
        "qid", "question", "answer", "answer_variant", "identity_display",
        "identity_category", "pair_id", "where", "role", "template"
    }

    def _load_row_level_if_valid(path: str):
        if not os.path.exists(path):
            return None
        try:
            df0 = pd.read_csv(path)
            if df0.empty:
                raise ValueError("empty CSV")
            if not need_cols.issubset(df0.columns):
                raise ValueError(f"bad schema: has {set(df0.columns)}")
            print(f"[RESUME] using existing row_level → {path} (rows={len(df0)})")
            return df0
        except Exception as e:
            print(f"[RESUME] invalid row_level at {path}: {e}; rebuilding…")
            return None

    df = None
    if not no_resume:
        df = _load_row_level_if_valid(row_path)
    else:
        if debug:
            print(f"[RESUME] Skipping resume due to --no-resume; will rebuild {row_path}")

    if debug and df is not None:
        got_q = df["qid"].nunique() if "qid" in df else None
        if isinstance(sample_n_questions, int) and got_q is not None and got_q != sample_n_questions:
            print(f"[DBG][WARN] Resumed row_level has questions={got_q}, which != requested --sample-nq={sample_n_questions}.")

    if df is None:
        records = []
        rng = np.random.default_rng(SEED)

        # originals
        for i in tqdm(idxs, desc="Building variants"):
            ex = ds[i]
            q = get_question(ex); a = choose_answer_sentence(ex)
            if not (q and a):
                continue
            records.append(dict(
                qid=i, question=q, answer_variant="original",
                entity="", label="", identity_display="",
                identity_category="", template="",
                role="original", where="", pair_id="", answer=a
            ))
            if len(records) >= max_row_level:
                break

        # identity + random injections
        if pairs_per_question > 0 and len(records) < max_row_level:
            pair_counter = 0
            for i in tqdm(idxs, desc="Identity & random injections"):
                if len(records) >= max_row_level:
                    break
                ex = ds[i]
                q = get_question(ex); a = choose_answer_sentence(ex)
                if not (q and a):
                    continue
                local_pairs = pick_pairs_for_question(pairs, pairs_per_question, rng)
                for ent, lab, cat in local_pairs:
                    disp, _ = realize_identity_ref(ent, lab, cat)
                    if "begin" in positions:
                        for tname in INJ_TEMPLATES_BEGIN.keys():
                            pair_counter += 1
                            pid = f"q{i}_b_{tname}_{pair_counter}"
                            # identity
                            records.append(dict(
                                qid=i, question=q, answer_variant="inject_begin",
                                entity=ent, label=lab, identity_display=disp,
                                identity_category=cat, template=tname,
                                role="identity", where="begin", pair_id=pid,
                                answer=inject_with_template(a, disp, "begin", tname)
                            ))
                            if len(records) >= max_row_level: break
                            # random
                            rand_phrase = make_random_phrase(rng)
                            records.append(dict(
                                qid=i, question=q, answer_variant="inject_begin",
                                entity="", label="", identity_display=rand_phrase,
                                identity_category="RANDOM_CONTROL", template=tname,
                                role="random", where="begin", pair_id=pid,
                                answer=inject_with_template(a, rand_phrase, "begin", tname)
                            ))
                            if len(records) >= max_row_level: break
                        if len(records) >= max_row_level: break
                    if "end" in positions and len(records) < max_row_level:
                        for tname in INJ_TEMPLATES_END.keys():
                            pair_counter += 1
                            pid = f"q{i}_e_{tname}_{pair_counter}"
                            # identity
                            records.append(dict(
                                qid=i, question=q, answer_variant="inject_end",
                                entity=ent, label=lab, identity_display=disp,
                                identity_category=cat, template=tname,
                                role="identity", where="end", pair_id=pid,
                                answer=inject_with_template(a, disp, "end", tname)
                            ))
                            if len(records) >= max_row_level: break
                            # random
                            rand_phrase = make_random_phrase(rng)
                            records.append(dict(
                                qid=i, question=q, answer_variant="inject_end",
                                entity="", label="", identity_display=rand_phrase,
                                identity_category="RANDOM_CONTROL", template=tname,
                                role="random", where="end", pair_id=pid,
                                answer=inject_with_template(a, rand_phrase, "end", tname)
                            ))
                            if len(records) >= max_row_level: break
                        if len(records) >= max_row_level: break

        df = pd.DataFrame(records)
        tmp = row_path + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, row_path)

    # --------------------------
    # Embeddings
    # --------------------------
    qid_to_q = df.groupby("qid")["question"].first().to_dict()
    q_texts = _truncate_list([qid_to_q[k] for k in qid_to_q])
    a_texts = _truncate_list(df["answer"].tolist())

    q_vecs = batch_embed_any(q_texts, model, batch_size=embed_batch_size,
                             normalize=True, progress=progress, desc="Embeddings: questions")
    a_vecs = batch_embed_any(a_texts, model, batch_size=embed_batch_size,
                             normalize=True, progress=progress, desc="Embeddings: answers")

    qid_index = {qid: i for i, qid in enumerate(qid_to_q.keys())}
    row_q_idx = df["qid"].map(qid_index).to_numpy()
    q_mat = q_vecs[row_q_idx]; a_mat = a_vecs
    df["cos_sim"] = np.einsum("ij,ij->i", q_mat, a_mat)

    base = df[df["answer_variant"] == "original"].set_index("qid")["cos_sim"].to_dict()
    df["delta_vs_original"] = df.apply(
        lambda r: (r["cos_sim"] - base[r["qid"]]) if r["answer_variant"] != "original" else 0.0,
        axis=1
    )

    # --------------------------
    # Cross-category ANOVA + Tukey
    # --------------------------
    try:

        sub_cc = df[
            (df.get("role","") == "identity") &
            (df.get("identity_category","") != "RANDOM_CONTROL") &
            (df["answer_variant"].isin(["inject_begin","inject_end"])) &
            df["delta_vs_original"].notna()
        ].copy()

        cat_counts = sub_cc["identity_category"].value_counts()
        keep = cat_counts[cat_counts >= 1].index
        sub_cc = sub_cc[sub_cc["identity_category"].isin(keep)]

        if not sub_cc.empty and sub_cc["identity_category"].nunique() >= 2:
            anova_model = ols("delta_vs_original ~ C(identity_category)", data=sub_cc).fit()
            anova_table = sm.stats.anova_lm(anova_model, typ=2)
            anova_csv = os.path.join(out_dir, "rq1_anova_identity_category.csv")
            anova_table.to_csv(anova_csv)

            cat_stats = (sub_cc.groupby("identity_category")["delta_vs_original"]
                            .agg(mean="mean", std="std", n="count"))
            cat_stats["cohens_d"] = cat_stats["mean"] / cat_stats["std"].replace(0, np.nan)
            cat_stats_csv = os.path.join(out_dir, "rq1_anova_category_stats.csv")
            cat_stats.to_csv(cat_stats_csv)

            tukey = pairwise_tukeyhsd(
                endog=sub_cc["delta_vs_original"].values,
                groups=sub_cc["identity_category"].values,
                alpha=0.05
            )
            tukey_df = pd.DataFrame(
                data=tukey._results_table.data[1:],
                columns=[c.lower().replace(" ", "_") for c in tukey._results_table.data[0]]
            )
            tukey_csv = os.path.join(out_dir, "rq1_anova_tukey_hsd.csv")
            tukey_df.to_csv(tukey_csv, index=False)

            with open(os.path.join(out_dir, "rq1_anova_summary.txt"), "w") as f:
                f.write("=== OLS ANOVA (Δ ~ identity_category) ===\n")
                f.write(str(anova_table))
                f.write("\n\n=== Group means/SD/n/Cohen d ===\n")
                f.write(cat_stats.to_string())
                f.write("\n\n=== Tukey HSD (pairwise) ===\n")
                f.write(str(tukey.summary()))

            # --- ANOVA plots ---
            import matplotlib.pyplot as plt
            plt.style.use("default")

            # (1) Means ± CI
            fig, ax = plt.subplots(figsize=(10,6))
            se = cat_stats["std"] / np.sqrt(cat_stats["n"].clip(lower=1))
            ci = 1.96 * se
            ax.bar(cat_stats.index, cat_stats["mean"], yerr=ci, capsize=4)
            ax.axhline(0, ls="--", c="k")
            ax.set_ylabel("Δ vs original (mean ±95% CI)")
            ax.set_title("ANOVA: Category means of Δ")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rq1_anova_means.png")); plt.close()

            # (2) Tukey HSD forest
            fig, ax = plt.subplots(figsize=(10,6))
            for i, row in tukey_df.iterrows():
                ax.errorbar(
                    x=row["meandiff"], y=i,
                    xerr=[[row["meandiff"]-row["lower"]],[row["upper"]-row["meandiff"]]],
                    fmt="o", color="blue" if row["reject"] else "gray"
                )
            ax.axvline(0, ls="--", c="k")
            ax.set_yticks(range(len(tukey_df)))
            ax.set_yticklabels([f"{g1} vs {g2}" for g1,g2 in zip(tukey_df["group1"], tukey_df["group2"])])
            ax.set_xlabel("Mean difference (Δ)")
            ax.set_title("Tukey HSD pairwise comparisons")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rq1_anova_tukey.png")); plt.close()

            # (3) Violin plot
            fig, ax = plt.subplots(figsize=(10,6))
            sub_cc.boxplot(column="delta_vs_original", by="identity_category", ax=ax, grid=False)
            plt.xticks(rotation=45, ha="right")
            ax.axhline(0, ls="--", c="k")
            ax.set_ylabel("Δ vs original")
            ax.set_title("Distribution of Δ by identity category")
            plt.suptitle("")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rq1_anova_violin.png")); plt.close()

        else:
            print("[ANOVA] Skipped: not enough categories or data.")

    except Exception as e:
        print(f"[ANOVA] skipped → {e}")

    # ------------ identity vs random: Δ_norm ------------
    norm_rows = []
    if "pair_id" in df.columns:
        for pid, g in df[df["pair_id"] != ""].groupby("pair_id"):
            gi = g[g["role"] == "identity"]
            gr = g[g["role"] == "random"]
            if len(gi) == 1 and len(gr) == 1:
                di = float(gi["delta_vs_original"].iloc[0])
                dr = float(gr["delta_vs_original"].iloc[0])
                norm_rows.append(dict(
                    qid=int(gi["qid"].iloc[0]),
                    where=str(gi["where"].iloc[0]),
                    template=str(gi["template"].iloc[0]),
                    identity_display=str(gi["identity_display"].iloc[0]),
                    identity_category=str(gi["identity_category"].iloc[0]),
                    delta_identity=di,
                    delta_random=dr,
                    delta_norm=di - dr
                ))
    norm_df = pd.DataFrame(norm_rows)
    partial = os.path.join(out_dir, "rq1_norm_identity_vs_random.csv")
    norm_df.to_csv(partial + ".tmp", index=False); os.replace(partial + ".tmp", partial)

    norm_cat = (norm_df.groupby(["identity_category", "where"], as_index=False)
                      .agg(mean_norm=("delta_norm", "mean"),
                           n=("delta_norm", "count"))) if not norm_df.empty else pd.DataFrame()

    # ------------ identity vs identity (within category): Δ_diff ------------
    ivs_rows = []
    sub_ivi = df[(df["role"] == "identity") & (df["answer_variant"] != "original")] if "role" in df.columns else pd.DataFrame()
    if not sub_ivi.empty:
        for (qid, cat, where), g in sub_ivi.groupby(["qid", "identity_category", "answer_variant"]):
            gg = (g[["identity_display", "delta_vs_original"]]
                  .dropna()
                  .drop_duplicates(subset=["identity_display"]))
            if len(gg) >= 2:
                gg2 = gg.sort_values("identity_display").head(2)
                a_, b_ = gg2.iloc[0], gg2.iloc[1]
                ivs_rows.append(dict(
                    qid=qid, identity_category=cat, where=where,
                    id_a=a_["identity_display"], id_b=b_["identity_display"],
                    delta_a=float(a_["delta_vs_original"]), delta_b=float(b_["delta_vs_original"]),
                    delta_diff=float(a_["delta_vs_original"] - b_["delta_vs_original"])
                ))
    id_vs_id_df = pd.DataFrame(ivs_rows)

    # ------------ Aggregations & stats ------------
    inj_mask = df["answer_variant"] != "original"
    agg_identity = (
        df[inj_mask]
        .groupby(["identity_display", "identity_category", "answer_variant"], as_index=False)
        .agg(mean_delta=("delta_vs_original", "mean"),
             std_delta=("delta_vs_original", "std"),
             n=("delta_vs_original", "count"))
    ) if inj_mask.any() else pd.DataFrame(columns=["identity_display","identity_category","answer_variant","mean_delta","std_delta","n"])
    if not agg_identity.empty:
        for col in ["mean_delta", "std_delta", "n"]:
            agg_identity[col] = pd.to_numeric(agg_identity[col], errors="coerce")
        se = agg_identity["std_delta"] / np.sqrt(agg_identity["n"].where(agg_identity["n"] > 1))
        agg_identity["ci_lo"] = agg_identity["mean_delta"] - 1.96 * se
        agg_identity["ci_hi"] = agg_identity["mean_delta"] + 1.96 * se

    sub_pos = df[df["answer_variant"].isin(["inject_begin","inject_end"])]
    if not sub_pos.empty:
        pos_tbl = (
            sub_pos.pivot_table(
                index=["qid","identity_display","identity_category"],
                columns="answer_variant",
                values="delta_vs_original",
                aggfunc="mean"
            )
            .reset_index()
        )
    else:
        pos_tbl = pd.DataFrame()

    if not pos_tbl.empty and {"inject_begin","inject_end"}.issubset(pos_tbl.columns):
        pos_tbl["begin_minus_end"] = pos_tbl["inject_begin"] - pos_tbl["inject_end"]
        position_summary = (
            pos_tbl.groupby(["identity_display","identity_category"], as_index=False)
            .agg(mean_begin_minus_end=("begin_minus_end","mean"),
                 std=("begin_minus_end","std"),
                 n=("begin_minus_end","count"))
            .sort_values("mean_begin_minus_end", ascending=False)
        )
        from scipy import stats as _stats
        def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
            d = x - y
            if d.size < 2: return np.nan
            sd = d.std(ddof=1)
            return d.mean() / sd if sd > 0 else np.nan
        paired_rows = []
        for disp, g in pos_tbl.groupby("identity_display"):
            x = g["inject_begin"].to_numpy(float)
            y = g["inject_end"].to_numpy(float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[mask], y[mask]
            if x.size >= 2:
                t, p = _stats.ttest_rel(x, y)
                paired_rows.append(dict(
                    identity_display=disp,
                    n=int(x.size),
                    t_stat=float(t),
                    p_value=float(p),
                    cohens_d=float(cohens_d_paired(x, y))
                ))
        paired_begin_end_stats = pd.DataFrame(paired_rows) if paired_rows else pd.DataFrame(columns=["identity_display","n","t_stat","p_value","cohens_d"])
    else:
        position_summary = pd.DataFrame(columns=["identity_display","identity_category","mean_begin_minus_end","std","n"])
        paired_begin_end_stats = pd.DataFrame(columns=["identity_display","n","t_stat","p_value","cohens_d"])

    # One-sample tests on Δ vs 0
    tests = []
    sub = df[df["answer_variant"].isin(["inject_begin","inject_end"])]
    if not sub.empty:
        from scipy import stats as _stats
        for (disp, var), g in sub.groupby(["identity_display","answer_variant"]):
            x = g["delta_vs_original"].to_numpy(float)
            x = x[~np.isnan(x)]
            if x.size >= 2:
                t, p = _stats.ttest_1samp(x, 0.0)
                tests.append(dict(
                    identity_display=disp,
                    answer_variant=var,
                    n=int(x.size),
                    mean_delta=float(np.mean(x)),
                    std_delta=float(np.std(x, ddof=1)),
                    t_stat=float(t),
                    p_value=float(p),
                ))
    ttest_by_identity = pd.DataFrame(tests) if tests else pd.DataFrame(columns=["identity_display","answer_variant","n","mean_delta","std_delta","t_stat","p_value"])

    # Qualitative: top drift examples
    tmp = df[df["answer_variant"] != "original"].copy()
    if not tmp.empty:
        tmp["abs_delta"] = tmp["delta_vs_original"].abs()
        top_per_q = tmp.loc[tmp.groupby("qid")["abs_delta"].idxmax(), :]
        base_df = df[df["answer_variant"] == "original"][["qid","answer"]].rename(columns={"answer":"original_answer"})
        question_top_drift = (
            top_per_q.merge(df[["qid","question"]].drop_duplicates(), on="qid", how="left")
                     .merge(base_df, on="qid", how="left")
                     .sort_values("abs_delta", ascending=False)
                     .head(qualitative_top_k)
        )
    else:
        question_top_drift = pd.DataFrame(columns=["qid","question","original_answer","answer_variant","identity_display","abs_delta"])

    # Save category norm agg early
    if out_dir and not norm_cat.empty:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "rq1_norm_by_category.csv")
        norm_cat.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
    
        # --------------------------
    # Per-category coverage (unique identities)
    # --------------------------
    try:
        sub_cov = df[(df["role"] == "identity") & (df["identity_category"] != "RANDOM_CONTROL")]
        per_cat_identities = (
            sub_cov.groupby("identity_category")["identity_display"].nunique().reset_index()
        )
        per_cat_identities.columns = ["identity_category", "unique_identities_used"]

        cov_path = os.path.join(out_dir, "rq1_per_category_identities.csv")
        per_cat_identities.to_csv(cov_path, index=False)
        print(f"[SAVE] per-category identities → {cov_path}")

        # Optional: quick bar plot
        plt.figure(figsize=(10,6))
        plt.bar(per_cat_identities["identity_category"], per_cat_identities["unique_identities_used"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Unique identities used")
        plt.title("RQ1: Per-category unique identity counts")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rq1_per_category_identities.png"))
        plt.close()
        print(f"[PLOT:RQ1:COV] wrote → rq1_per_category_identities.png")
    except Exception as e:
        print(f"[COV] skipped → {e}")


    return {
        "row_level": df,
        "agg_identity": agg_identity,
        "position_summary": position_summary,
        "paired_begin_end_stats": paired_begin_end_stats,
        "ttest_by_identity": ttest_by_identity,
        "question_top_drift": question_top_drift,
        "norm_identity_vs_random": norm_df,
        "norm_by_category": norm_cat,
        "identity_vs_identity": id_vs_id_df
    }

# ============================================
# Summary
# ============================================
def summarize_rq1(results: Dict[str, pd.DataFrame]):
        # sanity check, Returns a one-line summary (counts of identities, questions, total rows) from the results dict

    row_df = results.get("row_level")
    if row_df is None or row_df.empty:
        return "RQ1: no data"

    # ALL identity_display (includes originals and randoms)
    identities_all = row_df["identity_display"].nunique()

    # TRUE identities only (role=='identity' & not RANDOM_CONTROL)
    mask_true = (row_df.get("role","") == "identity") & (row_df.get("identity_category","") != "RANDOM_CONTROL")
    identities_true = row_df.loc[mask_true, "identity_display"].nunique()

    n_questions  = row_df["qid"].nunique() if "qid" in row_df else None
    n_rows = len(row_df)

    return (f"RQ1 → identities_all={identities_all}, identities_true={identities_true}, "
            f"questions={n_questions}, total_rows={n_rows}")

# ====================================================
# Main / CLI
# ====================================================
def main():
    # make the script a reproducible CLI tool.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-embedding-30m-english") # select ai model
    ap.add_argument("--out-dir", default="./out") # define output directly, /out as catch all
    ap.add_argument("--seed", type=int, default=42) # reproducibility 

    # RQ1 controls
    ap.add_argument("--sample-nq", type=int, default=200, help="Number of NQ questions to sample")
    ap.add_argument("--per-category", type=int, default=50, help="Identities per BBQ category to harvest")
    ap.add_argument("--begin-and-end", action="store_true", help="Inject at both begin and end")
    ap.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for embedding calls")


    # RITS Specific
    ap.add_argument("--use-rits-embed", action="store_true", help="Use RITS embeddings instead of sentence-transformers") 
    ap.add_argument("--rits-key", type=str, default=os.getenv("RITS_API_KEY"), help="RITS API key (or set env RITS_API_KEY)")
    ap.add_argument("--rits-embed-model", type=str, default="ibm/slate-125m-english-rtrvr-v2", help="RITS embedding model name")


    ## avoid memory limits 
    ap.add_argument("--rits-timeout", type=float, default=300.0, help="Per-request timeout (seconds) for RITS")
    ap.add_argument("--rits-max-retries", type=int, default=6, help="Max retries per RITS request")
    ap.add_argument("--rits-min-batch", type=int, default=4, help="Smallest batch when auto-splitting on timeouts")
    ap.add_argument("--rits-sleep", type=float, default=0.15, help="Sleep seconds between RITS batches")
    ap.add_argument("--pairs-per-question", type=int, default=10,
                    help="Max identity pairs sampled per question (prevents cartesian blowup)")
    
    # more limits 
    ap.add_argument("--allow-huge", action="store_true",
                    help="Allow runs that would exceed ~10M rows")
    ap.add_argument("--embed-max-chars", type=int, default=0,
                    help="Hard cap on text length (characters) before embedding; 0=off")
    ap.add_argument("--embed-max-tokens", type=int, default=0,
                    help="Approx token cap (regex-based) before embedding; 0=off")
    # ap.add_argument("--rits-endpoint", type=str, default=None,
    #                 help="Override RITS model endpoint (skip discovery)") # suggested by LLM, never worked?
    ap.add_argument("--max-row-level", type=int, default=1_1000_000_000,
                    help="Hard cap on total rows in rq1_row_level to prevent blow-ups")


    # --- Debug & resume controls, suggestion---
    ap.add_argument("--debug-rq1", action="store_true",
                    help="Verbose phase logging + write debugging diagnostics.") # check for csv for verifying outputs
    ap.add_argument("--no-resume", action="store_true",
                help="Ignore existing rq1_row_level.csv and rebuild row-level from scratch.") # force rebuilds 

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Embedder
    model = load_embedder_from_args(args)

    # Load Granite if present (for identities)
    granite = pd.DataFrame()
    try:
        granite = load_granite(args.out_dir) # searches out directory for granite
        if not granite.empty:
            try:
                print("[DEBUG] example parsed answer_info:", granite["answer_info"].iloc[0])
                print("[DEBUG] example parsed add_meta:",   granite["additional_metadata"].iloc[0])
            except Exception:
                pass
            print("[LOAD] granite dtypes:", granite.dtypes.to_dict())
            print(f"[LOAD] Loaded existing granite.* → {granite.shape}")
        else:
            print("[LOAD] granite is empty.")
    except Exception as e:
        print(f"[LOAD] granite not found → {e}")

    try:
        print("[NQ] Loading a Natural Questions split...")
        nq = load_nq_any()

        results = run_rq1_study1(
            nq_dataset=nq,
            granite_df=granite,
            model=model,
            sample_n_questions=args.sample_nq,
            per_category=args.per_category,
            begin_and_end=args.begin_and_end,
            qualitative_top_k=20,
            progress=True,
            out_dir=args.out_dir,
            embed_batch_size=args.embed_batch_size,
            pairs_per_question=args.pairs_per_question,
            max_row_level=args.max_row_level,
            embed_max_chars=args.embed_max_chars,
            embed_max_tokens=args.embed_max_tokens,
            debug=args.debug_rq1,           
            no_resume=args.no_resume,      
        )
        # Save outputs
        for key, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                path = os.path.join(args.out_dir, f"rq1_{key}.csv")
                df.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
                print(f"[SAVE] {key} → {path} (rows={len(df)})")

        # Plots
        try:
            row_csv = os.path.join(args.out_dir, "rq1_row_level.csv")
            norm_csv = os.path.join(args.out_dir, "rq1_norm_identity_vs_random.csv")
            ivi_csv  = os.path.join(args.out_dir, "rq1_identity_vs_identity.csv")
            if os.path.exists(row_csv):
                plot_rq1_by_category_mean(
                    row_path=row_csv,
                    out_path=os.path.join(args.out_dir, "rq1_category_means.png"),
                )
            if os.path.exists(norm_csv):
                plot_rq1_template_effects(
                    norm_path=norm_csv,
                    out_path=os.path.join(args.out_dir, "rq1_template_effects.png"),
                )
                plot_rq1_identity_vs_random(
                    norm_path=norm_csv,
                    out_path=os.path.join(args.out_dir, "rq1_identity_vs_random.png"),
                )
            if os.path.exists(ivi_csv):
                plot_rq1_identity_vs_identity(
                    ivs_path=ivi_csv,
                    out_path=os.path.join(args.out_dir, "rq1_identity_vs_identity.png"),
                )
        except Exception as e:
            print(f"[PLOT:RQ1] skipped → {e}")

        print(summarize_rq1(results))
    except Exception as e:
        print(f"[RQ1] failed → {e}")

if __name__ == "__main__":
    main()
