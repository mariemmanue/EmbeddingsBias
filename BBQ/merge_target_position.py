#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from bbq_core import prepare_df_from_hf

# --------------------------- utils ---------------------------

def clean_rows_df(df: pd.DataFrame) -> pd.DataFrame:
    # convert common string "nan" variants to NA and drop all-empty rows
    df = df.replace({"NAN": pd.NA, "nan": pd.NA, "NaN": pd.NA, "": pd.NA})
    df = df.dropna(how="all")
    return df

def _ensure_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _out_path_for(in_path: str, out_suffix: str) -> str:
    p = Path(in_path)
    # insert suffix before extension
    return str(p.with_name(p.stem + out_suffix + p.suffix))

# --------------------------- main merge ---------------------------

def merge_one(
    rows_path: str,
    base_df: pd.DataFrame,
    out_suffix: str,
    overwrite: bool,
) -> Optional[str]:
    rows_df = pd.read_csv(rows_path)
    rows_df = clean_rows_df(rows_df)

    # required join keys
    need = ["category", "question_index", "example_id"]
    missing = [c for c in need if c not in rows_df.columns]
    if missing:
        print(f"[SKIP] {rows_path}: missing columns {missing}")
        return None

    # normalize join keys
    rows_df = _ensure_str(rows_df, need)
    base_keys = base_df[need + ["target_position", "target_entity_text", "other_entity_text",
                               "target_char_start", "other_char_start"]].copy()
    base_keys = _ensure_str(base_keys, need)

    out_path = _out_path_for(rows_path, out_suffix)
    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {out_path} exists (use --overwrite)")
        return out_path

    merged = rows_df.merge(base_keys, on=need, how="left", validate="m:1")

    # coverage checks
    cov_pos = merged["target_position"].notna().mean() if "target_position" in merged.columns else 0.0
    cov_tcs = merged["target_char_start"].notna().mean() if "target_char_start" in merged.columns else 0.0

    print(f"[MERGE] {Path(rows_path).name}: rows={len(rows_df)} "
          f"target_position_cov={cov_pos:.3f} target_char_start_cov={cov_tcs:.3f}")

    merged.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}")
    return out_path

def build_parser():
    ap = argparse.ArgumentParser(
        description="Merge target_position + target_entity_text + char offsets into existing __rows*.csv files."
    )
    ap.add_argument("--rows-glob", required=True, help='Glob for rows CSVs, e.g. "output/*__rows*.csv"')
    ap.add_argument("--out-suffix", default="__withpos", help='Suffix inserted before .csv (default="__withpos")')
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    ap.add_argument("--dataset-id", required=True, help="HF dataset id, e.g. heegyu/BBQ")
    ap.add_argument("--metadata-csv", required=True, help="Path to additional_metadata.csv")
    ap.add_argument("--hf-revision", default=None)

    # optional: restrict processing
    ap.add_argument("--only-contains", default=None,
                    help='Only process files whose path contains this substring (e.g. "google__embeddinggemma")')
    return ap

def main():
    args = build_parser().parse_args()

    # resolve input files
    paths = sorted(glob.glob(args.rows_glob))
    if args.only_contains:
        paths = [p for p in paths if args.only_contains in p]

    print(f"[INFO] rows_glob={args.rows_glob}")
    print(f"[INFO] matched_files={len(paths)}")
    if not paths:
        print("[EXIT] No files matched your --rows-glob.")
        return 0

    # build base df (this already includes target_position etc in your pipeline)
    print("[BASE] Regenerating base df via prepare_df_from_hf")
    base_df = prepare_df_from_hf(
        dataset_id=args.dataset_id,
        metadata_csv_path=args.metadata_csv,
        revision=args.hf_revision,
    )
    print(f"[BASE] rows={len(base_df)}")

    ok = 0
    for rp in paths:
        try:
            outp = merge_one(rp, base_df, args.out_suffix, args.overwrite)
            if outp:
                ok += 1
        except Exception as e:
            print(f"[ERROR] {rp}: {e}", file=sys.stderr)

    print(f"[DONE] wrote {ok}/{len(paths)} files")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
