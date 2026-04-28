#!/usr/bin/env python3
"""
Replace all missing values with 0 and write a new CSV.

- Treats NaN/None and blank strings as missing.
- Optionally replaces +/-inf with NaN before filling.
- Works in streaming mode for large CSVs.

Usage:
  python fill_missing.py \\
    --in merged_quantiles_all_with_elev.csv \\
    --out merged_quantiles_all_with_elev_filled.csv

For very large files:
  python fill_missing.py \\
    --in huge.csv --out huge_filled.csv --chunksize 200000
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from wem.utils.logging import log


def process_df(df: pd.DataFrame, numeric_only: bool, zero_str: bool, replace_inf: bool) -> pd.DataFrame:
    # Treat empty strings as NaN so they get filled too
    df = df.replace(r"^\s*$", np.nan, regex=True)

    if replace_inf:
        df = df.replace([np.inf, -np.inf], np.nan)

    if numeric_only:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        if zero_str:
            # if requested, also turn NaNs in non-numeric into "0"
            other = [c for c in df.columns if c not in num_cols]
            if other:
                df[other] = df[other].fillna("0")
    else:
        df = df.fillna(0)

    return df

def main():
    ap = argparse.ArgumentParser(description="Replace missing values with 0 and write a new CSV.")
    ap.add_argument("--in",  dest="infile",  type=Path, required=True, help="Input CSV or Parquet")
    ap.add_argument("--out", dest="outfile", type=Path, required=True, help="Output CSV")
    ap.add_argument("--chunksize", type=int, default=0, help="CSV streaming chunk size (rows). 0 = read all at once.")
    ap.add_argument("--numeric-only", action="store_true",
                    help="Only fill numeric columns with 0 (leave non-numeric NaNs alone unless --zero-str is set).")
    ap.add_argument("--zero-str", action="store_true",
                    help='When --numeric-only is used, also fill missing in non-numeric columns with the string "0".')
    ap.add_argument("--no-inf-fix", dest="replace_inf", action="store_false",
                    help="Do not convert +/-inf to NaN before filling (default converts).")
    ap.set_defaults(replace_inf=True)
    args = ap.parse_args()

    if args.outfile.suffix.lower() != ".csv":
        log("[WARN] Output is not .csv; writing CSV anyway.")

    ext = args.infile.suffix.lower()
    if ext == ".parquet":
        log(f"Loading Parquet: {args.infile}")
        df = pd.read_parquet(args.infile)
        log("Filling missing values with 0 ...")
        df = process_df(df, numeric_only=args.numeric_only, zero_str=args.zero_str, replace_inf=args.replace_inf)
        log(f"Writing CSV -> {args.outfile}")
        df.to_csv(args.outfile, index=False)
    else:
        if args.chunksize and args.chunksize > 0:
            log(f"Streaming CSV in chunks of {args.chunksize:,} rows: {args.infile}")
            header_written = False
            total = 0
            for chunk in pd.read_csv(args.infile, chunksize=args.chunksize, low_memory=False):
                chunk = process_df(chunk, numeric_only=args.numeric_only, zero_str=args.zero_str, replace_inf=args.replace_inf)
                mode = "w" if not header_written else "a"
                chunk.to_csv(args.outfile, mode=mode, header=not header_written, index=False)
                header_written = True
                total += len(chunk)
            log(f"Wrote {total:,} rows -> {args.outfile}")
        else:
            log(f"Loading CSV: {args.infile}")
            df = pd.read_csv(args.infile, low_memory=False)
            log("Filling missing values with 0 ...")
            df = process_df(df, numeric_only=args.numeric_only, zero_str=args.zero_str, replace_inf=args.replace_inf)
            log(f"Writing CSV -> {args.outfile}")
            df.to_csv(args.outfile, index=False)

    log("Done.")

if __name__ == "__main__":
    main()
