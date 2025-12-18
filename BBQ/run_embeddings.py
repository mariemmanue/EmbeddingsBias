import os
import argparse
import json
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from bbq_core import (
    load_df_any,
    prepare_df_from_hf,
    Embedder,
    choices_from_ans_fields,
    VALID_LETTERS,
)

# ----------------------------- CLI -----------------------------

EXP_TYPES = ["qa_vs_context", "question_vs_context", "answer_vs_question"]

def build_parser():
    ap = argparse.ArgumentParser(
        description="Run EMBEDDING-only BBQ experiments and append CSVs in chunks."
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--append-chunk-size", type=int, default=200, help="Write to CSV every N *examples*.")

    ap.add_argument(
        "--exp-type",
        default="qa_vs_context",
        choices=EXP_TYPES,
        help=(
            "Which text-pairing to run: "
            "qa_vs_context=(Q+A) vs Context; "
            "question_vs_context=Question vs Context; "
            "answer_vs_question=Answer option vs Question."
        ),
    )

    ap.add_argument("--category", help="Single category to filter to.")
    ap.add_argument("--categories", help="Comma-separated list of categories.")
    ap.add_argument("--embedding-models", nargs="*", default=[
        "ibm-granite/granite-embedding-small-english-r2",
        "Qwen/Qwen3-Embedding-4B",
        "google/embeddinggemma-300m",
    ])

    ap.add_argument("--save-embs", action="store_true", help="Persist normalized query/doc vectors to disk per chunk.")
    ap.add_argument("--save-deltas", action="store_true", help="Also save delta vectors (doc - query) per chunk.")
    ap.add_argument("--emb-format", default="parquet", choices=["npy", "csv", "parquet"], help="Format for saved embeddings.")

    ap.add_argument("--df-path", help="Path to premerged BBQ dataframe.")
    ap.add_argument("--dataset-id", help="HF dataset id for BBQ.")
    ap.add_argument("--metadata-csv", help="Path to additional_metadata.csv")
    ap.add_argument("--hf-revision", default=None)
    return ap

# ----------------------------- UTIL -----------------------------

def _save_embeddings(df, embs, path, format, include_metadata=True):
    """
    Save embeddings in CSV, Parquet, or NumPy (.npy).
    If include_metadata is True and df is provided, saves embeddings with metadata columns.
    """
    if include_metadata and df is not None and len(df) == len(embs):
        emb_df = df.copy()
        for dim_idx in range(embs.shape[1]):
            emb_df[f"emb_{dim_idx}"] = embs[:, dim_idx]

        if format == "csv":
            emb_df.to_csv(path, index=False)
        elif format == "parquet":
            emb_df.to_parquet(path, index=False)
        else:
            np.save(path, embs)
            meta_path = path.replace(".npy", "__metadata.csv")
            df.to_csv(meta_path, index=False)
    else:
        if format == "csv":
            pd.DataFrame(embs).to_csv(path, index=False)
        elif format == "parquet":
            pd.DataFrame(embs).to_parquet(path, index=False)
        else:
            np.save(path, embs)

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
    print(f"[FILTER] Requested {sorted(list(target_cats))}, got {len(filtered)} rows (of {len(df)})")
    return filtered

def _pair_texts_for_row(
    row_dict: Dict[str, Any],
    example_idx: int,
    exp_type: str,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Returns: query_texts, doc_texts, meta_rows for ONE example.
    """
    question = str(row_dict.get("question", "")).strip()
    context = str(row_dict.get("context", "")).strip()
    choices = choices_from_ans_fields(row_dict)
    n_choices = len(choices)

    query_texts: List[str] = []
    doc_texts: List[str] = []
    meta_rows: List[Dict[str, Any]] = []

    base_meta = {
        "idx": example_idx,
        "category": row_dict.get("category", None),
        "example_id": row_dict.get("example_id", None),
        "question_index": row_dict.get("question_index", None),
        "question_polarity": row_dict.get("question_polarity", None),
        "label": row_dict.get("label", None),
        "target_loc": row_dict.get("target_loc", None),
        "question": question,
        "context": context,
        "exp_type": exp_type,
    }

    if exp_type == "question_vs_context":
        # One pair per example: Q vs Context
        # Keep schema consistent with answer-based modes via sentinels.
        query_texts.append(question)
        doc_texts.append(context)
        m = dict(base_meta)
        m.update({
            "pair_type": "Q_vs_C",
            "answer_idx": -1,
            "answer_letter": "",
            "answer_text": "",
            # also keep the QA field present (empty) so downstream code can rely on it
            "question_with_answer": "",
            "query_text": question,
            "doc_text": context,
        })
        meta_rows.append(m)
        return query_texts, doc_texts, meta_rows


    if exp_type == "answer_vs_question":
        # Three pairs per example: Answer option vs Question (ARC-style)
        for ans_idx in range(n_choices):
            ans_text = str(choices[ans_idx]).strip()
            query_texts.append(ans_text)
            doc_texts.append(question)
            m = dict(base_meta)
            m.update({
                "pair_type": "A_vs_Q",
                "answer_idx": ans_idx,
                "answer_letter": VALID_LETTERS[ans_idx],
                "answer_text": ans_text,
                "query_text": ans_text,
                "doc_text": question,
            })
            meta_rows.append(m)
        return query_texts, doc_texts, meta_rows

    # default: qa_vs_context (current)
    for ans_idx in range(n_choices):
        ans_text = str(choices[ans_idx]).strip()
        q_text = f"{question} {ans_text}".strip()
        query_texts.append(q_text)
        doc_texts.append(context)
        m = dict(base_meta)
        m.update({
            "pair_type": "QA_vs_C",
            "answer_idx": ans_idx,
            "answer_letter": VALID_LETTERS[ans_idx],
            "answer_text": ans_text,
            "question_with_answer": q_text,
            "query_text": q_text,
            "doc_text": context,
        })
        meta_rows.append(m)
    return query_texts, doc_texts, meta_rows

# ----------------------------- MAIN RUNNER -----------------------------

def run_embeddings_for_model(
    model_name: str,
    df_with_texts: pd.DataFrame,
    out_dir: str,
    device: str,
    dtype: str,
    batch_size: int,
    exp_type: str,
    append_chunk_size: int = 200,
    save_embs: bool = False,
    save_deltas: bool = False,
    emb_format: str = "parquet",
) -> None:
    """
    Runs one chosen experiment type for one model; appends chunked output to CSV.

    exp_type:
      - qa_vs_context: (Question + Answer option) vs Context   [CURRENT]
      - question_vs_context: Question vs Context              [Context vs Question]
      - answer_vs_question: Answer option vs Question         [ARC-style]
    """
    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.replace("/", "__")

    rows_path = os.path.join(out_dir, f"{safe}__rows__{exp_type}.csv")
    meta_path = os.path.join(out_dir, f"{safe}__emb_meta__{exp_type}.json")
    emb_index_path = os.path.join(out_dir, f"{safe}__emb_index__{exp_type}.csv")

    print(f"\n[EMB] Loading embedder: {model_name}")
    emb = Embedder(model_name, device=device, dtype=dtype, batch_size=batch_size)

    # Resume by example idx
    start_idx = 0
    if os.path.exists(rows_path):
        try:
            existing = pd.read_csv(rows_path, usecols=["idx"])
            if len(existing) > 0:
                start_idx = int(existing["idx"].max()) + 1
            print(f"[EMB] Resuming {model_name} ({exp_type}) from example idx {start_idx}")
        except Exception:
            print(f"[EMB] Existing rows file unreadable; removing and restarting: {rows_path}")
            try:
                os.remove(rows_path)
            except Exception:
                pass
            start_idx = 0

    n_total = len(df_with_texts)
    print(f"[EMB] Total examples: {n_total}")

    n_done_pairs = 0
    dim_q = None
    dim_c = None

    for i in range(start_idx, n_total, append_chunk_size):
        base_chunk = df_with_texts.iloc[i : i + append_chunk_size].copy()
        if base_chunk.empty:
            continue

        query_texts: List[str] = []
        doc_texts: List[str] = []
        meta_rows: List[Dict[str, Any]] = []

        for local_row_idx, (_, row) in enumerate(base_chunk.iterrows()):
            example_idx = i + local_row_idx
            row_dict = row.to_dict()

            qts, dts, mrs = _pair_texts_for_row(row_dict, example_idx, exp_type)
            query_texts.extend(qts)
            doc_texts.extend(dts)
            meta_rows.extend(mrs)

        if not query_texts:
            continue

        # Encode
        q_vecs = emb.encode_queries(query_texts, batch_size=batch_size, normalize=True)
        c_vecs = emb.encode_docs(doc_texts, batch_size=batch_size, normalize=True)

        # With normalize=True, cosine similarity is dot product
        sims = (q_vecs * c_vecs).sum(axis=1)

        records = []
        for meta_row, sim_val in zip(meta_rows, sims):
            rec = meta_row.copy()
            rec["sim"] = float(sim_val)
            rec["model_name"] = model_name
            records.append(rec)

        out_df = pd.DataFrame(records)

        # Append rows
        header = not os.path.exists(rows_path)
        out_df.to_csv(rows_path, mode="a", header=header, index=False)

        # Save vectors if requested
        if save_embs:
            base = os.path.join(out_dir, f"{safe}__{exp_type}__chunk_{i:08d}_{i+len(base_chunk)-1}")
            q_path = base + "__q." + emb_format
            c_path = base + "__c." + emb_format

            _save_embeddings(out_df, q_vecs, q_path, emb_format, include_metadata=True)
            _save_embeddings(out_df, c_vecs, c_path, emb_format, include_metadata=True)

            d_path = ""
            if save_deltas:
                deltas = c_vecs - q_vecs
                d_path = base + "__d." + emb_format
                _save_embeddings(out_df, deltas, d_path, emb_format, include_metadata=True)

            idx_row = pd.DataFrame([{
                "model_name": model_name,
                "exp_type": exp_type,
                "row_start_example": i,
                "row_end_example": i + len(base_chunk) - 1,
                "n_pairs": len(out_df),
                "q_path": q_path,
                "c_path": c_path,
                "d_path": d_path,
            }])
            idx_header = not os.path.exists(emb_index_path)
            idx_row.to_csv(emb_index_path, mode="a", header=idx_header, index=False)

        n_done_pairs += len(out_df)
        dim_q = q_vecs.shape[1]
        dim_c = c_vecs.shape[1]

        print(f"[EMB] {model_name} ({exp_type}): wrote {len(out_df)} pairs "
              f"for examples {i}–{i+len(base_chunk)-1}")

    if n_done_pairs > 0:
        meta = {
            "model_name": model_name,
            "exp_type": exp_type,
            "n_pairs": n_done_pairs,
            "q_dim": int(dim_q) if dim_q is not None else None,
            "c_dim": int(dim_c) if dim_c is not None else None,
            "note": {
                "qa_vs_context": "Each example expanded into one query per answer option: (question + answer) vs context.",
                "question_vs_context": "Each example produces one pair: question vs context.",
                "answer_vs_question": "Each example expanded into one pair per answer option: answer_text vs question.",
            }.get(exp_type, ""),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[WRITE] Embedding meta -> {meta_path}")

    print(f"[DONE] {model_name} ({exp_type}) embedding run.")

# ----------------------------- ENTRY -----------------------------

def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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

    if args.category or args.categories:
        df = _filter_by_categories(df, args.category, args.categories)
        if df.empty:
            print("[EXIT] No rows after filtering.")
            return

    if args.device == "mps" and args.dtype == "bfloat16":
        print("[WARN] MPS does not like bfloat16; using float16.")
        args.dtype = "float16"
    if args.device == "cpu" and args.dtype != "float32":
        print("[WARN] CPU best with float32; switching.")
        args.dtype = "float32"

    for model_name in args.embedding_models:
        run_embeddings_for_model(
            model_name=model_name,
            df_with_texts=df,
            out_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            exp_type=args.exp_type,
            append_chunk_size=args.append_chunk_size,
            save_embs=args.save_embs,
            save_deltas=args.save_deltas,
            emb_format=args.emb_format,
        )

if __name__ == "__main__":
    main()
