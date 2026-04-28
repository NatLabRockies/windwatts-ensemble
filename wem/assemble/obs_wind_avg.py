#!/usr/bin/env python3
# make_site_height_wsavg.py
"""
From a long-form training table (one row per quantile 'qnum'),
compute the mean wind speed per (station_id, height_m).

Input (ml_training_data.csv):
  station_id,name,lat,lon,height_m,qnum,observation,...,elevation_m,...

Output:
  station_id,name,lat,lon,elevation_m,height_m,ws_avg

Mean = integral_0^1 Q(p) dp using trapezoidal rule over qnum in [0..100].
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def first_nonnull(s: pd.Series):
    for v in s:
        if pd.notna(v):
            return v
    return np.nan


def mean_from_qnum_block(qnum: np.ndarray, obs: np.ndarray) -> float:
    """Integrate observation vs normalized quantile p=qnum/qmax."""
    if qnum.size < 2 or obs.size < 2:
        return np.nan
    # sort by qnum, drop NaNs in obs
    ord_ = np.argsort(qnum)
    q = qnum[ord_].astype("float64")
    y = obs[ord_].astype("float64")
    good = np.isfinite(q) & np.isfinite(y)
    q, y = q[good], y[good]
    if q.size < 2:
        return np.nan
    # normalize to [0,1] using the group's max qnum
    qmax = np.nanmax(q)
    if not np.isfinite(qmax) or qmax <= 0:
        return np.nan
    p = q / qmax
    # trapezoidal integral over p
    try:
        m = float(np.trapezoid(y, x=p))
    except AttributeError:
        m = float(np.trapz(y, x=p))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", type=Path, required=True, help="combined_quantiles_long_with_topo_loocv_10km.csv")
    ap.add_argument("--out", dest="outfile", type=Path, required=True, help="site_height_ws_avg.csv")
    ap.add_argument("--min-rows", type=int, default=5, help="Minimum qnum points required per group")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, low_memory=False)

    # Types
    for c in ["lat", "lon", "height_m", "qnum", "observation", "elevation_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    req_cols = ["station_id", "name", "lat", "lon", "height_m", "qnum", "observation", "elevation_m"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    rows = []
    gcols = ["station_id", "height_m"]
    for (sid, z), g in df.groupby(gcols, dropna=False):
        if len(g) < args.min_rows:
            ws_avg = np.nan
        else:
            ws_avg = mean_from_qnum_block(g["qnum"].to_numpy(), g["observation"].to_numpy())

        rows.append({
            "station_id": sid,
            "name": first_nonnull(g["name"]),
            "lat": first_nonnull(g["lat"]),
            "lon": first_nonnull(g["lon"]),
            "elevation_m": first_nonnull(g["elevation_m"]),
            "height_m": z,
            "ws_avg": ws_avg,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.outfile, index=False)
    print(f"Wrote {len(out):,} rows → {args.outfile}")

if __name__ == "__main__":
    main()
