#!/usr/bin/env python
import os
import re
import argparse
import pandas as pd

"""
Merge chunked parquet embedding files per model into single files.

We expect filenames like:
  Qwen__Qwen3-Embedding-4B__chunk_00056000_56199__q.parquet
  ibm-granite__granite-embedding-small-english-r2__chunk_00026800_26999__c.parquet

This script will create:
  Qwen__Qwen3-Embedding-4B__q.parquet
  Qwen__Qwen3-Embedding-4B__c.parquet
  Qwen__Qwen3-Embedding-4B__d.parquet
  ...etc

and (optionally) delete the original chunk_* files.
"""

PATTERN = re.compile(
    r"^(?P<prefix>.+)__chunk_(?P<start>\d+)_(?P<end>\d+)__(?P<typ>[qcd])\.parquet$"
)

def find_groups(output_dir: str):
    groups = {}
    for fname in os.listdir(output_dir):
        m = PATTERN.match(fname)
        if not m:
            continue
        prefix = m.group("prefix")
        typ = m.group("typ")  # 'q', 'c', or 'd'
        key = (prefix, typ)
        groups.setdefault(key, []).append(fname)
    return groups

def merge_group(output_dir: str, prefix: str, typ: str, files: list[str],
                overwrite: bool = False, delete_chunks: bool = False):
    files = sorted(files)  # chunk_0000..., chunk_0200..., etc.
    out_path = os.path.join(output_dir, f"{prefix}__{typ}.parquet")

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {out_path} already exists; use --overwrite to re-create.")
        return

    dfs = []
    for f in files:
        full = os.path.join(output_dir, f)
        print(f"[READ] {full}")
        dfs.append(pd.read_parquet(full))

    merged = pd.concat(dfs, ignore_index=True)
    print(f"[WRITE] {out_path}  (rows={len(merged)})")
    merged.to_parquet(out_path)

    if delete_chunks:
        for f in files:
            full = os.path.join(output_dir, f)
            print(f"[DELETE] {full}")
            os.remove(full)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="Directory containing chunked parquet files.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing merged __{typ}.parquet files.")
    ap.add_argument("--delete-chunks", action="store_true",
                    help="Delete the original __chunk_* parquet files after merging.")
    ap.add_argument("--model-prefix", default=None,
                    help="Optional: only merge for this prefix (e.g. 'Qwen__Qwen3-Embedding-4B').")
    args = ap.parse_args()

    outdir = args.output_dir
    groups = find_groups(outdir)
    if not groups:
        print("[INFO] No chunked parquet files found.")
        return

    print(f"[INFO] Found {len(groups)} (model_prefix, typ) groups.")

    for (prefix, typ), files in groups.items():
        if args.model_prefix and prefix != args.model_prefix:
            continue
        print(f"\n[GROUP] prefix={prefix}, typ={typ}, n_files={len(files)}")
        merge_group(
            output_dir=outdir,
            prefix=prefix,
            typ=typ,
            files=files,
            overwrite=args.overwrite,
            delete_chunks=args.delete_chunks,
        )

    print("\n[DONE] Merge complete.")

if __name__ == "__main__":
    main()
