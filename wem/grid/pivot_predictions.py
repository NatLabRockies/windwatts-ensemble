#!/usr/bin/env python3
"""
Collapse per-(site,height,quantile) predictions into one row per (site,height)
with q000..q100 columns and attach ERA5 grid index.

Inputs:
  --in    predictions CSV/Parquet from infer_xgb.py
          (must have: lat, lon, height_m, qnum, pred_observation)
  --era5  era5_location_data.csv with columns: index, latitude, longitude

Output:
  --out   CSV/Parquet with columns:
          index, latitude, longitude, height_m, q000..q100  (predicted)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from wem.utils.logging import log
from wem.utils.io import read_table, write_table
from wem.constants import QCOLS


def main():
    ap = argparse.ArgumentParser(description="Build site-level predicted quantiles table.")
    ap.add_argument("--in",   dest="infile",  type=Path, required=True)
    ap.add_argument("--era5", dest="era5file",type=Path, required=True)
    ap.add_argument("--out",  dest="outfile", type=Path, required=True)
    ap.add_argument("--decimals", type=int, default=6,
                    help="Rounding used to match lat/lon to ERA5 grid (default: 6).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.outfile.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.outfile} (use --overwrite).")

    # ---- Load data
    log(f"Loading predictions: {args.infile}")
    df = read_table(args.infile)
    need_pred_cols = {"lat","lon","height_m","qnum","pred_observation"}
    missing = need_pred_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing columns: {sorted(missing)}")

    log(f"Loading ERA5 grid locations: {args.era5file}")
    era = read_table(args.era5file)
    need_era_cols = {"index","latitude","longitude"}
    miss_era = need_era_cols - set(era.columns)
    if miss_era:
        raise ValueError(f"ERA5 file missing columns: {sorted(miss_era)}")

    # ---- Clean types
    for c in ["lat","lon","height_m","qnum","pred_observation"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["latitude","longitude"]:
        era[c] = pd.to_numeric(era[c], errors="coerce")

    # Optional sanity: clamp qnum to 0..100 and integerize
    df = df[np.isfinite(df["qnum"])]
    df["qnum"] = df["qnum"].round().astype(int)
    df = df[(df["qnum"] >= 0) & (df["qnum"] <= 100)]

    # ---- Key for joining to ERA5 index: rounded coords
    r = int(args.decimals)
    df["_lat_r"]  = df["lat"].round(r)
    df["_lon_r"]  = df["lon"].round(r)
    era["_lat_r"] = era["latitude"].round(r)
    era["_lon_r"] = era["longitude"].round(r)

    # Drop NAs in coordinates
    df = df.dropna(subset=["_lat_r","_lon_r","height_m"]).copy()
    era = era.dropna(subset=["_lat_r","_lon_r"]).copy()

    # ---- Merge to attach ERA5 index & canonical lat/lon
    merged = df.merge(
        era[["index","latitude","longitude","_lat_r","_lon_r"]],
        on=["_lat_r","_lon_r"],
        how="left",
        validate="many_to_one",  # many pred rows to one ERA5 site
    )

    n_unmatched = int(merged["index"].isna().sum())
    if n_unmatched > 0:
        log(f"[WARN] {n_unmatched:,} prediction rows did not match an ERA5 site with rounding={r}. "
            "They will be dropped.")
        merged = merged.dropna(subset=["index"]).copy()

    # ---- Pivot to q000..q100 per (site,height)
    # If duplicates exist for a given qnum, take the mean.
    merged["qcol"] = merged["qnum"].astype(int).map(lambda q: f"q{q:03d}")
    # We pivot via pivot_table for stability
    pv = merged.pivot_table(
        index=["index","latitude","longitude","height_m"],
        columns="qcol",
        values="pred_observation",
        aggfunc="mean",
    ).reset_index()

    # Ensure all q000..q100 exist
    for qc in QCOLS:
        if qc not in pv.columns:
            pv[qc] = np.nan

    # Order columns
    pv = pv[["index","latitude","longitude","height_m"] + QCOLS].copy()

    # Sort nicely
    pv = pv.sort_values(["index","height_m"]).reset_index(drop=True)

    # ---- Save
    log(f"Writing output -> {args.outfile} (rows={len(pv):,})")
    write_table(pv, args.outfile)
    log("Done.")

if __name__ == "__main__":
    main()
