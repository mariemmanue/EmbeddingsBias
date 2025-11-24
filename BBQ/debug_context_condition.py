#!/usr/bin/env python3
"""
Debug script to check for discrepancies between computed context_condition_3
and any existing context condition field in the BBQ dataset.

Binary mapping:
  - HF "ambig"  <-> context_condition_3 == "AMBIG"
  - HF "disambig_*"  <-> context_condition_3 in {"DISAMBIG_STEREO", "DISAMBIG_ANTI"}

Additionally:
  - Rows where context_condition_3 == "DISAMBIG" are treated as suspicious and reported.

Usage:
    python debug_context_condition.py --dataset-id heegyu/BBQ --metadata-csv additional_metadata.csv
    python debug_context_condition.py --df-path path/to/prepared_dataframe.csv
"""

import argparse
import pandas as pd
from typing import Any

from bbq_core import (
    load_df_any,
    prepare_df_from_hf,
)


def build_parser():
    ap = argparse.ArgumentParser(
        description="Debug context_condition_3 discrepancies in BBQ dataset."
    )
    ap.add_argument("--df-path", help="Path to pre-prepared DataFrame (CSV or Parquet)")
    ap.add_argument("--dataset-id", help="HuggingFace dataset ID")
    ap.add_argument("--metadata-csv", help="Path to additional_metadata.csv")
    ap.add_argument("--hf-revision", default=None, help="Dataset revision")
    ap.add_argument(
        "--output",
        default="context_condition_discrepancies.csv",
        help="Output path for discrepancies CSV",
    )
    ap.add_argument(
        "--no-print",
        action="store_true",
        help="Don't print summary (only save CSV)",
    )
    return ap


def _norm_hf_cc(val: Any) -> str:
    """
    Normalize HF context_condition to a binary label:
      - "AMBIG" for any value containing 'ambig'
      - "DISAMBIG" for anything else non-empty
      - "" for empty / NaN
    """
    s = str(val).strip().lower()
    if not s or s in {"nan", "none"}:
        return ""
    if "disambig" in s:
        return "DISAMBIG"
    # treat all non-ambig contexts as disambiguated
    return "AMBIG"


def _norm_ours_cc(val: Any) -> str:
    """
    Normalize context_condition_3 to:
      - "AMBIG" if exactly "AMBIG"
      - "DISAMBIG" if DISAMBIG_STEREO or DISAMBIG_ANTI
      - "OTHER" for raw "DISAMBIG" or anything unexpected
    """
    s = str(val).strip().upper()
    if s == "AMBIG":
        return "AMBIG"
    if s in {"DISAMBIG_STEREO", "DISAMBIG_ANTI"}:
        return "DISAMBIG"
    # this catches plain "DISAMBIG" and any weird values
    return "OTHER"


def main():
    args = build_parser().parse_args()

    # Load DataFrame
    if args.df_path:
        df = load_df_any(args.df_path)
        print(f"[LOAD] Loaded {len(df)} rows from {args.df_path}")
    else:
        if not (args.dataset_id and args.metadata_csv):
            raise SystemExit(
                "Must provide either --df-path OR both --dataset-id and --metadata-csv"
            )
        df = prepare_df_from_hf(
            dataset_id=args.dataset_id,
            metadata_csv_path=args.metadata_csv,
            revision=args.hf_revision,
        )
        print(f"[LOAD] Prepared DataFrame with {len(df)} rows")

    # Check if context_condition_3 exists
    if "context_condition_3" not in df.columns:
        print("[ERROR] DataFrame missing 'context_condition_3' column.")
        print(f"Available columns: {sorted(df.columns.tolist())}")
        return

    # Try to find an HF context condition column
    orig_cc_col = None
    for c in ["context_condition", "condition", "cc"]:
        if c in df.columns:
            orig_cc_col = c
            break

    if orig_cc_col is None:
        print("[WARN] No HF context condition column found (context_condition/condition/cc).")
        print("[INFO] Will only report rows where context_condition_3 == 'DISAMBIG'.")
        df["ours_bin"] = df["context_condition_3"].map(_norm_ours_cc)
        discrepancies = df[df["ours_bin"] == "OTHER"].copy()
    else:
        # Normalize HF + our context conditions
        df["hf_bin"] = df[orig_cc_col].map(_norm_hf_cc)
        df["ours_bin"] = df["context_condition_3"].map(_norm_ours_cc)

        # 1) Mismatches between HF and our binary mapping
        bin_mismatches = df[
            (df["hf_bin"].isin({"AMBIG", "DISAMBIG"})) &
            (df["ours_bin"].isin({"AMBIG", "DISAMBIG"})) &
            (df["hf_bin"] != df["ours_bin"])
        ]

        # 2) Rows where ours_bin == "OTHER" (i.e., raw DISAMBIG or weird)
        other_weird = df[df["ours_bin"] == "OTHER"]

        # Union of both sources of “error”
        discrepancies = (
            pd.concat([bin_mismatches, other_weird], ignore_index=True)
            .drop_duplicates(subset=["category", "question_index", "example_id"])
        )

    # Save discrepancies
    if len(discrepancies) > 0:
        cols_to_show = [
            "category",
            "question_index",
            "example_id",
        ]
        if orig_cc_col is not None:
            cols_to_show += [orig_cc_col, "hf_bin"]
        cols_to_show += [
            "context_condition_3",
            "ours_bin",
            "label",
            "target_loc",
            "ans0",
            "ans1",
            "ans2",
        ]

        # Some columns may be missing depending on df, filter safely
        cols_to_show = [c for c in cols_to_show if c in discrepancies.columns]

        discrepancies[cols_to_show].to_csv(args.output, index=False)
    else:
        # still create an empty CSV for reproducibility
        discrepancies.to_csv(args.output, index=False)

    if not args.no_print:

        print("\n=== SUMMARY ===")
        if "hf_bin" in df.columns:
            print("HF context_condition (binary):")
            print(df["hf_bin"].value_counts(dropna=False).to_string())
            print("\nOur context_condition_3 (mapped to binary):")
            print(df["ours_bin"].value_counts(dropna=False).to_string())
        else:
            print("No HF context_condition column; only ours_bin available.")
            print("\nOur context_condition_3 (mapped to binary):")
            print(df["ours_bin"].value_counts(dropna=False).to_string())

        print(f"\n[RESULT] Found {len(discrepancies)} suspicious rows.")
        if len(discrepancies) > 0:
            print(f"[RESULT] Saved discrepancies to: {args.output}")
            print("\nSample of discrepancies:")
            print(discrepancies[cols_to_show].head(20).to_string(index=False))
        else:
            print("[SUCCESS] No discrepancies under the AMBIG/DISAMBIG mapping!")

if __name__ == "__main__":
    main()
