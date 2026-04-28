#!/usr/bin/env python3
"""
Merge HRRR / WTK / WTK-LED per-height quantile files into one wide CSV.

- Detects files named like:
    hrrr_quantiles_30m.csv
    wtk_quantiles_80m.csv
    wtk_led_quantiles_100m.csv
- Adds columns:
    source  (values: HRRR, WTK, WTK-LED)
    height  (integer height in meters parsed from filename)
  (keeps existing height_m from file; if missing/NaN, fills from parsed height)

- Quantile columns are auto-detected by regex ^q\\d{3,4}$ so both q000..q100 and q000..q1000 work.

Usage
-----
python merge_grid.py --in-dir ./quantiles_dir --out-file merged_quantiles_all.csv

Options
-------
--source-col   Name for the new source column (default: source)
--overwrite    Replace output if it exists (else append-safe error)
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from wem.utils.logging import log

RX_FILE = re.compile(r"^(hrrr|wtk|wtk_led)_quantiles_(\d+)m\.csv$", re.IGNORECASE)
RX_QCOL = re.compile(r"^q\d{3,4}$", re.IGNORECASE)

SOURCE_LABEL = {
    "hrrr": "HRRR",
    "wtk": "WTK",
    "wtk_led": "WTK-LED",
}

def find_input_files(in_dir: Path) -> List[Path]:
    return sorted([fp for fp in in_dir.glob("*_quantiles_*m.csv") if RX_FILE.match(fp.name)])

def detect_qcols(fp: Path) -> List[str]:
    # Read header only
    cols = pd.read_csv(fp, nrows=0).columns.tolist()
    qcols = [c for c in cols if RX_QCOL.match(str(c))]
    # sort by numeric value of the suffix
    def qkey(c: str) -> int:
        return int(re.sub(r"[^0-9]", "", c))
    return sorted(qcols, key=qkey)

def read_one(fp: Path, source_col: str) -> pd.DataFrame:
    m = RX_FILE.match(fp.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {fp.name}")
    src_key, z_str = m.group(1).lower(), m.group(2)
    src = SOURCE_LABEL[src_key]
    z_int = int(z_str)

    qcols = detect_qcols(fp)
    base_cols = ["grid_id", "lat", "lon", "height_m"]
    usecols = [c for c in base_cols if c in pd.read_csv(fp, nrows=0).columns] + qcols
    df = pd.read_csv(fp, usecols=lambda c: (c in usecols), low_memory=False)

    # Types & cleanup
    df["grid_id"] = df["grid_id"].astype("string")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    if "height_m" not in df.columns:
        df["height_m"] = float(z_int)
    else:
        df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce").fillna(float(z_int))

    # Extra columns
    df[source_col] = src
    df["height"] = int(z_int)

    # Column order: id, coords, source, heights, then q*
    cols_order = ["grid_id", "lat", "lon", source_col, "height_m", "height"] + qcols
    df = df[cols_order]

    # Drop obviously bad rows
    df = df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Merge HRRR/WTK/WTK-LED quantile CSVs into one file.")
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory with *_quantiles_*m.csv files.")
    ap.add_argument("--out-file", type=Path, required=True, help="Output CSV path.")
    ap.add_argument("--source-col", type=str, default="source", help="Name of the source column to add.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = ap.parse_args()

    args.in_dir = args.in_dir.resolve()
    args.out_file = args.out_file.resolve()

    files = find_input_files(args.in_dir)
    if not files:
        raise SystemExit(f"No files like '*_quantiles_*m.csv' found in {args.in_dir}")

    if args.out_file.exists():
        if args.overwrite:
            args.out_file.unlink()
        else:
            raise SystemExit(f"Output already exists: {args.out_file} (use --overwrite)")

    log(f"Found {len(files)} files. Merging -> {args.out_file.name}")

    header_written = False
    total_rows = 0
    # Use first file to lock column order (q000..qN)
    first_df = read_one(files[0], source_col=args.source_col)
    col_order = first_df.columns.tolist()
    first_df.to_csv(args.out_file, index=False)
    header_written = True
    total_rows += len(first_df)
    log(f"  - {files[0].name}: {len(first_df):,} rows")

    for fp in files[1:]:
        df = read_one(fp, source_col=args.source_col)
        # align columns (in case q-range differs)
        missing = [c for c in col_order if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        extra = [c for c in df.columns if c not in col_order]
        if extra:
            # New quantiles? extend order at the end (rare, but allowed)
            col_order += extra
            # rewrite header with new order by recreating file (safe approach for simplicity)
            old = pd.read_csv(args.out_file)
            old = old.reindex(columns=col_order)
            old.to_csv(args.out_file, index=False)

        df = df.reindex(columns=col_order)
        df.to_csv(args.out_file, mode="a", header=False, index=False)

        total_rows += len(df)
        log(f"  - {fp.name}: {len(df):,} rows")

    log(f"Done. Wrote total rows: {total_rows:,}")
    log(f"Output: {args.out_file}")

if __name__ == "__main__":
    main()
