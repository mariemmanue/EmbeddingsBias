import os
import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd
from bbq_core import (
    load_df_any,
    prepare_df_from_hf,
    Embedder,
)

"""
python run_embeddings.py --output-dir ./output \
                         --device auto \
                         --dtype float16 \
                         --batch-size 64 \
                         --append-chunk-size 200 \
                         --dataset-id heegyu/BBQ \
                         --metadata-csv ./data/additional_metadata.csv \
                         --embedding-models google/embeddinggemma-300m ibm-granite/granite-embedding-small-english-r2 Qwen/Qwen3-Embedding-4B \
                         --save-embs \
                         --save-deltas \
                         --emb-format parquet
"""

def build_parser():
    ap = argparse.ArgumentParser(
        description="Run EMBEDDING-only BBQ experiments and append CSVs in chunks."
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--append-chunk-size", type=int, default=200, help="Write to CSV every N rows.")
    ap.add_argument("--category", help="Single category to filter to.")
    ap.add_argument("--categories", help="Comma-separated list of categories.")
    ap.add_argument("--embedding-models", nargs="*", default=[
        "ibm-granite/granite-embedding-small-english-r2",
        "Qwen/Qwen3-Embedding-4B",
        "google/embeddinggemma-300m",
    ])
    ap.add_argument("--save-embs", action="store_true", help="Persist normalized question/context vectors to disk per chunk.")
    ap.add_argument("--save-deltas", action="store_true", help="Also save delta vectors (context - question) per chunk.")
    ap.add_argument("--emb-format", default="parquet", choices=["npy", "csv", "parquet"], help="Format for saved embeddings.")
    ap.add_argument("--df-path", help="Path to premerged BBQ dataframe.")
    ap.add_argument("--dataset-id", help="HF dataset id for BBQ.")
    ap.add_argument("--metadata-csv", help="Path to additional_metadata.csv")
    ap.add_argument("--hf-revision", default=None)
    return ap

def _save_embeddings(df, embs, path, format):
    # This function saves embeddings in the specified format: CSV, Parquet, or NumPy (.npy).
    if format == "csv":
        pd.DataFrame(embs).to_csv(path, index=False)
    elif format == "parquet":
        pd.DataFrame(embs).to_parquet(path, index=False)
    else:
        np.save(path, embs)

def _filter_by_categories(df: pd.DataFrame, cat_arg: Optional[str], cats_arg: Optional[str]) -> pd.DataFrame:
    if "category" not in df.columns: # If the "category" column is not present in the DataFrame, returns the DataFrame as is.
        return df
    if not cat_arg and not cats_arg: # If neither cat_arg nor cats_arg is provided, returns the DataFrame as is.
        return df
    df = df.copy()
    df["category"] = df["category"].astype(str)
    df["_cat_upper"] = df["category"].str.upper() # Converts the "category" column to uppercase for consistent comparison.
    target_cats = set()
    if cat_arg: # cat_arg: A single category to filter by (if provided).
        target_cats.add(cat_arg.strip().upper())
    if cats_arg: # cats_arg: A comma-separated list of categories to filter by (if provided).
        for c in cats_arg.split(","):
            if c.strip():
                target_cats.add(c.strip().upper())
    filtered = df[df["_cat_upper"].isin(target_cats)].drop(columns=["_cat_upper"])
    print(f"[FILTER] Requested {sorted(list(target_cats))}, got {len(filtered)} rows (of {len(df)})")
    return filtered

def _save_embeddings(df, embs, path, format):
    if format == "csv":
        pd.DataFrame(embs).to_csv(path, index=False)
    elif format == "parquet":
        pd.DataFrame(embs).to_parquet(path, index=False)
    else:  # default to .npy
        np.save(path, embs)

def run_embeddings_for_model(
    model_name: str,
    df_with_texts: pd.DataFrame,
    out_dir: str,
    device: str,
    dtype: str,
    batch_size: int,
    append_chunk_size: int = 200,
    save_embs: bool = False,
    save_deltas: bool = False,
    emb_format: str = "parquet",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.replace("/", "__")
    rows_path = os.path.join(out_dir, f"{safe}__rows.csv")
    meta_path = os.path.join(out_dir, f"{safe}__emb_meta.json")
    emb_index_path = os.path.join(out_dir, f"{safe}__emb_index.csv")

    print(f"\n[EMB] Loading embedder: {model_name}")
    emb = Embedder(model_name, device=device, dtype=dtype, batch_size=batch_size)

    # Figure out how many rows already processed
    start_idx = 0
    if os.path.exists(rows_path):
        try:
            existing = pd.read_csv(rows_path, usecols=["idx"])
            start_idx = existing["idx"].max() + 1
            print(f"[EMB] Resuming {model_name} from row {start_idx}")
        except Exception:
            print("[EMB] Existing file not readable; will overwrite.")
            start_idx = 0

    n_total = len(df_with_texts)
    print(f"[EMB] Total rows: {n_total}")

    # Running stats for meta
    n_done = 0
    q_norm_sum = 0.0
    c_norm_sum = 0.0
    dim_q = None
    dim_c = None

    # Main loop
    for i in range(start_idx, n_total, append_chunk_size):
        chunk_df = df_with_texts.iloc[i : i + append_chunk_size].copy()
        questions = chunk_df["question"].astype(str).tolist()
        contexts = chunk_df["context"].astype(str).tolist()

        q_vecs = emb.encode_queries(questions, batch_size=batch_size, normalize=True)
        c_vecs = emb.encode_docs(contexts, batch_size=batch_size, normalize=True)

        # Step 1: Calculate the dot products
        dot_products = (q_vecs * c_vecs).sum(axis=1)
        # Step 2: Calculate the norms of vectors
        norm_q = np.linalg.norm(q_vecs, axis=1)
        norm_c = np.linalg.norm(c_vecs, axis=1)
        # Step 3: Calculate cosine similarity
        sims = dot_products / (norm_q * norm_c)

        # Save rows CSV
        chunk_df = chunk_df.copy()
        chunk_df["idx"] = range(i, i + len(chunk_df))
        chunk_df["sim"] = sims
        chunk_df["model_name"] = model_name

        header = not os.path.exists(rows_path) or i == 0
        chunk_df.to_csv(rows_path, mode="a", header=header, index=False)

        # Save vectors
        if save_embs:
            base = os.path.join(out_dir, f"{safe}__chunk_{i:08d}_{i+len(chunk_df)-1}")
            q_path = base + "__q." + emb_format
            c_path = base + "__c." + emb_format
            _save_embeddings(chunk_df, q_vecs, q_path, emb_format)
            _save_embeddings(chunk_df, c_vecs, c_path, emb_format)

            d_path = None
            if save_deltas:
                deltas = c_vecs - q_vecs
                d_path = base + "__d." + emb_format
                _save_embeddings(chunk_df, deltas, d_path, emb_format)

            # Append an index row so we can reload later
            idx_row = pd.DataFrame([{
                "model_name": model_name,
                "row_start": i,
                "row_end": i + len(chunk_df) - 1,
                "q_path": q_path,
                "c_path": c_path,
                "d_path": d_path if d_path else ""
            }])
            idx_header = not os.path.exists(emb_index_path)
            idx_row.to_csv(emb_index_path, mode="a", header=idx_header, index=False)

        q_norm_sum += float(np.linalg.norm(q_vecs, axis=1).sum())
        c_norm_sum += float(np.linalg.norm(c_vecs, axis=1).sum())
        n_done += len(chunk_df)
        dim_q = q_vecs.shape[1]
        dim_c = c_vecs.shape[1]

        print(f"[EMB] {model_name}: wrote rows {i}–{i+len(chunk_df)-1}")

    if n_done > 0:
        meta = {
            "model_name": model_name,
            "n_rows": n_done,
            "q_dim": int(dim_q) if dim_q is not None else None,
            "c_dim": int(dim_c) if dim_c is not None else None,
            "q_norm_mean": q_norm_sum / n_done,
            "c_norm_mean": c_norm_sum / n_done,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[WRITE] Embedding meta -> {meta_path}")

    print(f"[DONE] {model_name} embedding run.")

    if n_done > 0:
        meta = {
            "model_name": model_name,
            "n_rows": n_done,
            "q_dim": int(dim_q) if dim_q is not None else None,
            "c_dim": int(dim_c) if dim_c is not None else None,
            "q_norm_mean": q_norm_sum / n_done,
            "c_norm_mean": c_norm_sum / n_done,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[WRITE] Embedding meta -> {meta_path}")

    print(f"[DONE] {model_name} embedding run.")

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
            append_chunk_size=args.append_chunk_size,
            save_embs=args.save_embs,
            save_deltas=args.save_deltas,
            emb_format=args.emb_format,
        )

if __name__ == "__main__":
    main()