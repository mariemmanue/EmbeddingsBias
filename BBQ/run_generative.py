# run_generative.py
import os
import argparse
from typing import Optional, List

import pandas as pd
import torch

from bbq_core import (
    load_df_any,
    prepare_df_from_hf,
    choices_from_ans_fields,
)


def first_letter(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()
    for ch in t:
        if ch in ("A", "B", "C"):
            return ch
    return None


def race_prompt_paper(question, choices, context):
    lines = [context.strip(), question]
    for i, ch in enumerate("abc"[: len(choices)]):
        lines.append(f"({ch}) {choices[i]}")
    return "\n".join(lines)


def arc_prompt_paper(context: str, question: str, choices: List[str]) -> str:
    first_line = f"{context.strip()}{question}"
    lines = [first_line]
    for i, ch in enumerate("abc"[: len(choices)]):
        lines.append(f"({ch}) {choices[i]}")
    return "\n".join(lines)


def _build_inputs_for_prompt(tool, prompt: str, device):
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
        return {k: v.to(device) for k, v in inputs.items()}


def score_next_token_logprobs(model, tool, prompt, device, letters):
    inputs = _build_inputs_for_prompt(tool, prompt, device)
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

    tok = getattr(tool, "tokenizer", tool)
    scores = {}
    for letter in letters:
        variants = [
            letter,
            " " + letter,
            letter + ")",
            "(" + letter + ")",
            letter + ".",
            letter + ":",
        ]
        ids = []
        for v in variants:
            try:
                toks = tok(v, add_special_tokens=False).input_ids
                if len(toks) == 1:
                    ids.append(toks[0])
            except Exception:
                pass
        if ids:
            scores[letter] = float(torch.max(logp[0, ids]))
    return scores


def _filter_by_categories(df: pd.DataFrame, cat_arg: Optional[str], cats_arg: Optional[str]) -> pd.DataFrame:
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
    print(f"[FILTER] requested {sorted(list(target_cats))}, got {len(filtered)} rows (of {len(df)})")
    return filtered


def evaluate_model_generative_stream(
    df: pd.DataFrame,
    model_name: str,
    out_dir: str,
    device: str = "auto",
    dtype: str = "float16",
    use_logprobs: bool = True,
    subset: Optional[int] = None,
    append_chunk_size: int = 200,
) -> None:
    # device
    if device == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype if device in ("cuda", "mps") else None,
        device_map="auto",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.replace("/", "__")
    rows_path = os.path.join(out_dir, f"{safe}__rows.csv")

    work_df = df if subset is None else df.head(subset)
    n_total = len(work_df)

    start_idx = 0
    if os.path.exists(rows_path):
        try:
            existing = pd.read_csv(rows_path, usecols=["idx"])
            start_idx = existing["idx"].max() + 1
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

            choices = [str(r.get(k, "")).strip() for k in ["ans0", "ans1", "ans2"] if str(r.get(k, "")).strip()]
            if len(choices) < 2:
                continue
            letters = list("ABC")[: len(choices)]

            gold = (str(r.get("gold_label", "")) or "").strip().upper()[:1]
            sc_l = (str(r.get("sc_label", "")) or "").strip().upper()[:1] or None
            si_l = (str(r.get("si_label", "")) or "").strip().upper()[:1] or None

            p_race = race_prompt_paper(question, choices, context)
            p_arc = arc_prompt_paper(context, question, choices)

            if use_logprobs:
                scores_arc = score_next_token_logprobs(model, tok, p_arc, device, letters)
                scores_race = score_next_token_logprobs(model, tok, p_race, device, letters)
                pred_arc = max(scores_arc.items(), key=lambda kv: kv[1])[0] if scores_arc else None
                pred_race = max(scores_race.items(), key=lambda kv: kv[1])[0] if scores_race else None
            else:
                scores_arc, scores_race = {}, {}
                # greedy gen
                inputs_arc = _build_inputs_for_prompt(tok, p_arc, device)
                with torch.no_grad():
                    gen_arc = model.generate(**inputs_arc, max_new_tokens=2, do_sample=False, temperature=0.0)
                cut_arc = inputs_arc["input_ids"].shape[1]
                dec_arc = tok.decode(gen_arc[0][cut_arc:], skip_special_tokens=True)
                pred_arc = first_letter(dec_arc)

                inputs_race = _build_inputs_for_prompt(tok, p_race, device)
                with torch.no_grad():
                    gen_race = model.generate(**inputs_race, max_new_tokens=2, do_sample=False, temperature=0.0)
                cut_race = inputs_race["input_ids"].shape[1]
                dec_race = tok.decode(gen_race[0][cut_race:], skip_special_tokens=True)
                pred_race = first_letter(dec_race)

            acc_arc = 1 if pred_arc and gold and pred_arc.upper() == gold.upper() else 0
            acc_race = 1 if pred_race and gold and pred_race.upper() == gold.upper() else 0

            qp = str(r.get("question_polarity", r.get("polarity", ""))).strip().upper().replace("-", "")
            cc3 = r.get("context_condition_3", r.get("context_condition", ""))

            rows_out.append(
                {
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
            )

        if rows_out:
            chunk_df = pd.DataFrame(rows_out)
            header = not os.path.exists(rows_path) or i == 0
            chunk_df.to_csv(rows_path, mode="a", header=header, index=False)
            print(f"[GEN] {model_name}: wrote rows {i}–{i+len(rows_out)-1}")

    print(f"[DONE] {model_name}: generative streaming complete. CSV at {rows_path}")
    

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
    )


if __name__ == "__main__":
    main()
