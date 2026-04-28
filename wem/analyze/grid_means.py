"""Compute mean wind speed from predicted quantiles.

Input : site_quantiles_predicted.(csv|parquet) with columns:
        index, latitude, longitude, height_m, q000..q100
Output: site_mean_winds.csv with columns:
        lat, lon, height_m, mean_ms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wem.constants import QCOLS
from wem.utils.logging import log
from wem.utils.quantiles import mean_from_quantiles


def main():
    ap = argparse.ArgumentParser(
        description="Compute mean wind (m/s) per site-height from predicted "
        "quantiles."
    )
    ap.add_argument(
        "--in",
        dest="infile",
        type=Path,
        required=True,
        help="site_quantiles_predicted.(csv|parquet)",
    )
    ap.add_argument(
        "--out",
        dest="outfile",
        type=Path,
        default=Path("site_mean_winds.csv"),
        help="Output CSV with lat, lon, height_m, mean_ms",
    )
    args = ap.parse_args()

    # --- Load & clean ---
    log(f"Loading predictions: {args.infile}")
    if args.infile.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(args.infile)
    else:
        df = pd.read_csv(args.infile, low_memory=False)

    # Rename latitude/longitude -> lat/lon and keep required cols
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise SystemExit(
            "Input must contain 'latitude' and 'longitude' columns."
        )
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    need = ["lat", "lon", "height_m"] + QCOLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    df = df[need].copy()

    # Numeric + drop bad rows
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)

    # Mean wind (m/s)
    log("Computing mean wind (trapz over q000..q100)")
    df_out = df[["lat", "lon", "height_m"]].copy()
    df_out["mean_ms"] = mean_from_quantiles(df)

    # Save
    log(f"Writing → {args.outfile}")
    df_out.to_csv(args.outfile, index=False)
    log(f"Done. Rows: {len(df_out):,}")


if __name__ == "__main__":
    main()
