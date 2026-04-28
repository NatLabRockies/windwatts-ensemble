#!/usr/bin/env python3
"""
Prepare full-grid data for inference.

Input (CSV/Parquet):
  - Columns include: lat, lon, height_m, elevation_m, a dataset/source column,
    and quantiles q000..q100 (or q000..q1000).

Expected dataset names (case-insensitive, flexible):
  - "WTK CONUS" -> wtk
  - "HRRR CONUS" -> hrrr
  - "WTK-LED CONUS" -> wtk_led_conus
  You can also already have "wtk", "hrrr", "wtk_led_conus".

Output:
  - CSV/Parquet with columns:
      lat, lon, elevation_m, height_m, qnum, wtk, hrrr, wtk_led_conus
  - One row per site-height-quantile.

Usage:
  python prepare_inference.py \\
    --in merged_quantiles_all_with_elev.csv \\
    --out inference_table.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from wem.utils.logging import log
from wem.utils.io import read_table, write_table


# ----------------------------- main logic ------------------------------
def normalize_dataset_names(s: pd.Series) -> pd.Series:
    """Map various dataset labels to canonical column names."""
    def _norm(x: str) -> str:
        t = str(x).strip().lower()
        if "wtk-led" in t or "wtk_led" in t:
            return "wtk_led_conus"
        if "hrrr" in t:
            return "hrrr"
        if "wtk" in t:
            return "wtk"
        # fallbacks if input already uses canonical names
        if t in {"wtk_led_conus", "hrrr", "wtk"}:
            return t
        return t  # leave as-is (will be dropped later if unknown)
    return s.map(_norm)


def find_dataset_column(df: pd.DataFrame) -> str:
    candidates = ["dataset", "source", "model", "product"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find a dataset/source column. Expected one of: "
        f"{candidates}. Your input should have per-row dataset identity."
    )


def find_quantile_columns(cols: List[str]) -> List[str]:
    """Return columns matching qNNN or qNNNN (e.g., q000..q100 or q000..q1000)."""
    rx = re.compile(r"^q\d{3,4}$", flags=re.IGNORECASE)
    qcols = [c for c in cols if rx.match(c)]
    if not qcols:
        raise ValueError("No quantile columns found (expected columns named like q000..q100 or q000..q1000).")
    # keep natural order: q000, q001, ...
    qcols.sort(key=lambda c: int(c[1:]))
    return qcols


def melt_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    # Required base columns
    for c in ["lat", "lon", "height_m", "elevation_m"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    # Dataset column
    ds_col = find_dataset_column(df)
    df["_dataset"] = normalize_dataset_names(df[ds_col])

    # Keep only rows from the 3 expected datasets
    keep = df["_dataset"].isin(["wtk", "hrrr", "wtk_led_conus"])
    df = df.loc[keep].copy()
    if df.empty:
        raise SystemExit("After filtering datasets, no rows remain. Check your dataset/source values.")

    # Quantile columns
    qcols = find_quantile_columns(df.columns.tolist())

    # Minimal ID vars to carry through the melt
    id_vars = ["lat", "lon", "height_m", "elevation_m", "_dataset"]

    log(f"Melting {len(qcols)} quantile columns -> long format ...")
    long = df.melt(id_vars=id_vars, value_vars=qcols, var_name="qname", value_name="value")

    # qnum: 0..100 (or 0..100.0 if input provided q000..q1000)
    qid = long["qname"].str[1:].astype(int)
    if qid.max() > 100:
        long["qnum"] = (qid / 10.0).astype("float32")
    else:
        long["qnum"] = qid.astype("int16").astype("float32")  # keep as float32 downstream

    # Remove non-finite values early
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long[np.isfinite(long["value"])]

    # Deduplicate within (lat,lon,height,elev,qnum,dataset) by mean (safe and stable)
    gb_keys = ["lat", "lon", "elevation_m", "height_m", "qnum", "_dataset"]
    long = long.groupby(gb_keys, as_index=False)["value"].mean()

    # Pivot datasets to columns
    log("Pivoting datasets to columns ...")
    wide = long.pivot_table(
        index=["lat", "lon", "elevation_m", "height_m", "qnum"],
        columns="_dataset",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Ensure the three expected columns exist
    for name in ["wtk", "hrrr", "wtk_led_conus"]:
        if name not in wide.columns:
            wide[name] = np.nan

    # Column order
    wide = wide[["lat", "lon", "elevation_m", "height_m", "qnum", "wtk", "hrrr", "wtk_led_conus"]]

    # Sort for readability
    wide = wide.sort_values(["height_m", "lat", "lon", "qnum"]).reset_index(drop=True)

    # Types
    wide["lat"] = pd.to_numeric(wide["lat"], errors="coerce").astype("float64")
    wide["lon"] = pd.to_numeric(wide["lon"], errors="coerce").astype("float64")
    wide["height_m"] = pd.to_numeric(wide["height_m"], errors="coerce").astype("float32")
    wide["elevation_m"] = pd.to_numeric(wide["elevation_m"], errors="coerce").astype("float32")
    wide["qnum"] = pd.to_numeric(wide["qnum"], errors="coerce").astype("float32")
    for c in ["wtk", "hrrr", "wtk_led_conus"]:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")

    return wide


def main():
    ap = argparse.ArgumentParser(description="Transform merged/elevation data into inference rows.")
    ap.add_argument("--in",  dest="infile",  type=Path, required=True, help="CSV or Parquet with quantiles + dataset + elevation_m")
    ap.add_argument("--out", dest="outfile", type=Path, required=True, help="Output CSV or Parquet")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.outfile.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.outfile} (use --overwrite).")

    log(f"Loading: {args.infile}")
    df = read_table(args.infile)

    log("Transforming to (lat, lon, elevation_m, height_m, qnum, wtk, hrrr, wtk_led_conus) ...")
    out = melt_and_pivot(df)

    log(f"Writing -> {args.outfile}  (rows={len(out):,})")
    write_table(out, args.outfile)
    log("Done.")


if __name__ == "__main__":
    main()
