# run_generative.py
"""
Example SLURM runs:

nlprun -q jag -p standard -g 1 -r 200G -c 8 -n BBQ-gen-model-mistral7B -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/EmbeddingsBias/BBQ && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate embbias && \
   export HF_HOME=/nlp/scr/mtano/hf_home && \
   python run_generative.py \
     --output-dir ./output \
     --device auto \
     --dtype float16 \
     --gen-model mistralai/Mistral-7B-Instruct-v0.3 \
     --use-logprobs \
     --subset 100 \
     --append-chunk-size 200 \
     --dataset-id heegyu/BBQ \
     --metadata-csv additional_metadata.csv"

nlprun -q jag -p standard -g 1 -r 200G -c 8 -n BBQ-gen-model-gpt-oss-20b -o slurm_logs/%x-%j.out \
  "cd /nlp/scr/mtano/EmbeddingsBias/BBQ && \
   . /nlp/scr/mtano/miniconda3/etc/profile.d/conda.sh && \
   conda activate embbias && \
   export HF_HOME=/nlp/scr/mtano/hf_home && \
   python run_generative.py \
     --output-dir ./output \
     --device auto \
     --dtype float16 \
     --gen-model openai/gpt-oss-20b \
     --use-logprobs \
     --subset 100 \
     --append-chunk-size 200 \
     --dataset-id heegyu/BBQ \
     --metadata-csv additional_metadata.csv"
"""

import os
import argparse
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd

from bbq_core import (
    load_df_any,
    prepare_df_from_hf,
    choices_from_ans_fields,
)


# ----------------- MODEL LOADING -----------------

def load_model(
    model_id: str,
    device: str = "auto",
    requested_dtype: str = "float16",
    device_map: Optional[str] = None,
):
    """
    Load a causal LM with sensible device / dtype handling and return
    (tokenizer, model, resolved_device).
    """

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # ------------------------------------------------------------------
    # SPECIAL-CASE: openai/gpt-oss-20b bf16/Half MXFP4 bug
    # Force CPU + float32 to avoid the mixed-dtype MoE crash.
    # ------------------------------------------------------------------
    if "gpt-oss-20b" in model_id:
        print(
            "[MODEL] Detected openai/gpt-oss-20b → forcing device=cpu, "
            "dtype=float32 to avoid MXFP4 bf16/Half mismatch."
        )
        device = "cpu"
        requested_dtype = "float32"
        device_map = None
    # ------------------------------------------------------------------

    # Resolve dtype and device_map
    if device == "cuda":
        major = torch.cuda.get_device_capability(0)[0]
        if requested_dtype == "bfloat16" and major < 8:
            print("[WARN] GPU does not support bfloat16; switching to float16.")
            requested_dtype = "float16"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(requested_dtype, torch.float16)
        device_map = "auto" if device_map is None else device_map
    elif device == "mps":
        if requested_dtype == "bfloat16":
            print("[WARN] MPS does not support bfloat16; switching to float16.")
            requested_dtype = "float16"
        torch_dtype = torch.float16 if requested_dtype == "float16" else torch.float32
        device_map = None
    else:
        if requested_dtype != "float32":
            print("[WARN] CPU best with float32; switching.")
        requested_dtype = "float32"
        torch_dtype = torch.float32
        device_map = None

    print(
        f"[MODEL] Loading {model_id} on {device} "
        f"with dtype={requested_dtype} (torch_dtype={torch_dtype}), "
        f"device_map={device_map}"
    )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,          # <-- use dtype instead of torch_dtype=
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Pad/EOS safeguards
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.pad_token_id = tok.pad_token_id
        if gen_cfg.eos_token_id is None and tok.eos_token_id is not None:
            gen_cfg.eos_token_id = tok.eos_token_id

    if device_map is None:
        model = model.to(device)
    model.eval()
    return tok, model, device



# ----------------- PROMPTS & PARSING -----------------


def first_letter(text: str, allowed_letters: List[str]) -> Optional[str]:
    """
    Return the first character in `text` that is in allowed_letters (case-insensitive),
    normalized to uppercase. If none, return None.
    """
    if not text:
        return None
    t = str(text).strip()
    allowed_upper = {ch.upper() for ch in allowed_letters}
    for ch in t:
        up = ch.upper()
        if up in allowed_upper:
            return up
    return None


def race_prompt_paper(question: str, choices: List[str], context: str) -> str:
    """
    BBQ 'race' prompt: context + question + multi-choice list.
    Supports arbitrary number of choices, labeled (A), (B), (C), ...
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [str(context).strip(), str(question).strip()]
    for i in range(len(choices)):
        ch = letters[i]
        lines.append(f"({ch}) {choices[i]}")
    return "\n".join(lines)


def arc_prompt_paper(question: str, choices: List[str]) -> str:
    """
    BBQ 'ARC-style' prompt: question + multi-choice list.
    Supports arbitrary number of choices, labeled (A), (B), (C), ...
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [str(question).strip()]
    for i in range(len(choices)):
        ch = letters[i]
        lines.append(f"({ch}) {choices[i]}")
    return "\n".join(lines)


def _build_inputs_for_prompt(tool, prompt: str, device: str):
    """
    Build model inputs from a tokenizer / chat template object.
    Handles both chat-template tokenizers and plain text tokenizers.
    """
    if hasattr(tool, "apply_chat_template") and getattr(tool, "chat_template", None) is not None:
        messages = [{"role": "user", "content": prompt}]
        inputs = tool.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        return {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = tool(prompt, return_tensors="pt")
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        return {k: v.to(device) for k, v in inputs.items()}


def score_next_token_logprobs(model, tok, prompt: str, device: str, letters: List[str]):
    """
    Compute log-probabilities for the next token corresponding to each letter in `letters`.
    Uses several token variants per letter to be robust across tokenizers.
    """
    inputs = _build_inputs_for_prompt(tok, prompt, device)
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}

    model_dtype = next(model.parameters()).dtype
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                inputs[k] = v.to(device=device, dtype=model_dtype)
            else:
                inputs[k] = v.to(device)

    with torch.no_grad():
        out = model(**inputs)
        next_logits = out.logits[:, -1, :]
        logp = torch.log_softmax(next_logits, dim=-1)

    scores = {}
    for letter in letters:
        # Variants to catch different tokenizations
        variants = [
            letter,
            " " + letter,
            "\n" + letter,
            letter + ")",
            "(" + letter + ")",
            letter + ".",
            letter + ":",
        ]
        token_ids = []
        for v in variants:
            try:
                ids = tok(v, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    token_ids.append(ids[0])
            except Exception:
                pass
        if token_ids:
            scores[letter] = float(torch.max(logp[0, token_ids]))
    return scores


# ----------------- DATA FILTERING -----------------


def _filter_by_categories(
    df: pd.DataFrame,
    cat_arg: Optional[str],
    cats_arg: Optional[str],
) -> pd.DataFrame:
    if "category" not in df.columns:
        return df
    if not cat_arg and not cats_arg:
        return df
    df = df.copy()
    df["category"] = df["category"].astype(str)
    df["_cat_upper"] = df["category"].str.upper()
    target_cats = set()
    if cat_arg:
        target_cats.add(cat_arg.strip().upper())
    if cats_arg:
        for c in cats_arg.split(","):
            if c.strip():
                target_cats.add(c.strip().upper())
    filtered = df[df["_cat_upper"].isin(target_cats)].drop(columns=["_cat_upper"])
    print(
        f"[FILTER] requested {sorted(list(target_cats))}, "
        f"got {len(filtered)} rows (of {len(df)})"
    )
    return filtered


# ----------------- MAIN EVAL LOOP -----------------


def evaluate_model_generative_stream(
    df: pd.DataFrame,
    model_name: str,
    out_dir: str,
    device: str = "auto",
    dtype: str = "float16",
    use_logprobs: bool = True,
    subset: Optional[int] = None,
    append_chunk_size: int = 200,
    save_logprobs: bool = False,
) -> None:
    # Resolve device if "auto"
    if device == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    tok, model, device = load_model(
        model_id=model_name,
        device=device,
        requested_dtype=dtype,
        device_map=None,
    )

    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.replace("/", "__")
    rows_path = os.path.join(out_dir, f"{safe}__rows.csv")

    work_df = df if subset is None else df.head(subset)
    n_total = len(work_df)

    # Resume if a partial CSV already exists
    start_idx = 0
    if os.path.exists(rows_path):
        try:
            existing = pd.read_csv(rows_path, usecols=["idx"])
            start_idx = int(existing["idx"].max()) + 1
            print(f"[GEN] Resuming {model_name} from row {start_idx}")
        except Exception:
            start_idx = 0

    for i in range(start_idx, n_total, append_chunk_size):
        chunk = work_df.iloc[i : i + append_chunk_size].copy()
        rows_out = []

        for local_j, r in enumerate(chunk.to_dict(orient="records")):
            global_idx = i + local_j

            question = str(r.get("question", "")).strip()
            context = str(r.get("context", "")).strip()
            cat = str(r.get("category", "")).strip()
            qi = str(r.get("question_index", global_idx)).strip()

            choices = choices_from_ans_fields(r)
            if len(choices) < 2:
                continue

            # Letters for this item, e.g., ["A", "B", "C", "D"]
            letters = [chr(ord("A") + k) for k in range(len(choices))]

            gold = (str(r.get("gold_label", "")) or "").strip().upper()[:1]
            sc_l = (str(r.get("sc_label", "")) or "").strip().upper()[:1] or None
            si_l = (str(r.get("si_label", "")) or "").strip().upper()[:1] or None

            p_race = race_prompt_paper(question, choices, context)
            p_arc = arc_prompt_paper(question, choices)

            if use_logprobs:
                scores_arc = score_next_token_logprobs(model, tok, p_arc, device, letters)
                scores_race = score_next_token_logprobs(model, tok, p_race, device, letters)
                pred_arc = max(scores_arc, key=scores_arc.get) if scores_arc else None
                pred_race = max(scores_race, key=scores_race.get) if scores_race else None
            else:
                # Greedy generation: read first allowed letter from completion
                # ARC
                inputs_arc = _build_inputs_for_prompt(tok, p_arc, device)
                with torch.no_grad():
                    gen_arc = model.generate(
                        **inputs_arc,
                        max_new_tokens=2,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id,
                    )
                cut_arc = inputs_arc["input_ids"].shape[1]
                dec_arc = tok.decode(gen_arc[0][cut_arc:], skip_special_tokens=True)
                pred_arc = first_letter(dec_arc, letters)

                # RACE
                inputs_race = _build_inputs_for_prompt(tok, p_race, device)
                with torch.no_grad():
                    gen_race = model.generate(
                        **inputs_race,
                        max_new_tokens=2,
                        do_sample=False,
                        temperature=0.0,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.pad_token_id,
                    )
                cut_race = inputs_race["input_ids"].shape[1]
                dec_race = tok.decode(gen_race[0][cut_race:], skip_special_tokens=True)
                pred_race = first_letter(dec_race, letters)

            acc_arc = 1 if pred_arc and gold and pred_arc.upper() == gold.upper() else 0
            acc_race = 1 if pred_race and gold and pred_race.upper() == gold.upper() else 0

            qp = str(r.get("question_polarity", r.get("polarity", ""))).strip().upper().replace("-", "")
            cc3 = r.get("context_condition_3", r.get("context_condition", ""))

            row = {
                "idx": global_idx,
                "category": cat,
                "question_index": qi,
                "question_polarity": qp,
                "context_condition_3": cc3,
                "choices": " ||| ".join(choices),
                "question": question,
                "context": context,
                "gold_label": gold,
                "sc_label": sc_l,
                "si_label": si_l,
                "prompt_arc": p_arc,
                "prompt_race": p_race,
                "pred_arc": pred_arc,
                "pred_race": pred_race,
                "acc_arc": acc_arc,
                "acc_race": acc_race,
                "model_name": model_name,
            }

            # Optionally store raw logprobs per letter
            if save_logprobs and use_logprobs:
                for letter in letters:
                    row[f"logp_arc_{letter}"] = scores_arc.get(letter)
                    row[f"logp_race_{letter}"] = scores_race.get(letter)

            rows_out.append(row)


        if rows_out:
            chunk_df = pd.DataFrame(rows_out)
            header = not os.path.exists(rows_path) or start_idx == 0 and i == 0
            chunk_df.to_csv(rows_path, mode="a", header=header, index=False)
            print(f"[GEN] {model_name}: wrote rows {i}–{i + len(rows_out) - 1}")

    print(f"[DONE] {model_name}: generative streaming complete. CSV at {rows_path}")


# ----------------- CLI -----------------


def build_parser():
    ap = argparse.ArgumentParser(
        description="Run GENERATIVE-only BBQ experiments, append every N rows, and resume."
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--gen-model", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    ap.add_argument("--use-logprobs", action="store_true")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--append-chunk-size", type=int, default=200)
    # data
    ap.add_argument("--df-path", help="Path to premerged BBQ dataframe.")
    ap.add_argument("--dataset-id", help="HF dataset id for BBQ.")
    ap.add_argument("--metadata-csv", help="Path to additional_metadata.csv")
    ap.add_argument("--hf-revision", default=None)
    ap.add_argument("--category", help="Filter to single category")
    ap.add_argument("--save-logprobs", action="store_true",
                help="Save per-letter logprob columns in the output CSV (requires --use-logprobs).")
    ap.add_argument("--categories", help="Comma-separated list of categories")
    return ap


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    if args.df_path:
        df = load_df_any(args.df_path)
        print(f"[DATA] Loaded {len(df)} rows from {args.df_path}")
    else:
        if not (args.dataset_id and args.metadata_csv):
            raise SystemExit("Must pass either --df-path OR both --dataset-id and --metadata-csv")
        df = prepare_df_from_hf(
            dataset_id=args.dataset_id,
            metadata_csv_path=args.metadata_csv,
            revision=args.hf_revision,
        )
        print(f"[DATA] Prepared merged HF+metadata dataframe with {len(df)} rows.")

    df = _filter_by_categories(df, args.category, args.categories)
    if df.empty:
        print("[EXIT] No rows after filtering.")
        return

    # dtype/device safeguards
    if args.device == "mps" and args.dtype == "bfloat16":
        print("[WARN] MPS + bfloat16 → switching to float16.")
        args.dtype = "float16"
    if args.device == "cpu" and args.dtype != "float32":
        print("[WARN] CPU best with float32 → switching.")
        args.dtype = "float32"

    evaluate_model_generative_stream(
        df=df,
        model_name=args.gen_model,
        out_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        use_logprobs=args.use_logprobs,
        subset=args.subset,
        append_chunk_size=args.append_chunk_size,
        save_logprobs=args.save_logprobs,
    )



if __name__ == "__main__":
    main()
