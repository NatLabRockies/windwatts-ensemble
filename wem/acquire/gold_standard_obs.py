#!/usr/bin/env python3
"""
Gold-standard observations -> per-site quantiles (q000..q100), 2007-2024 only.

Changes:
  * Do NOT round timestamps and do NOT de-duplicate samples.
  * Keep years in [2007, 2024].
  * Month-balance filter per (site_id, height, year):
      - At least MIN_MONTHS months with >0 samples, AND
      - At least MIN_MONTHS months with count >= MIN_FRAC_OF_MEDIAN * median(nonzero monthly count).
  * Write a CSV listing filtered-out site-years with diagnostics and reasons.

Output schema matches resource datasets:
  station_id,name,lat,lon,elev_m,dataset,height_m,interp,agg,years,processed_utc,q000..q100

Migrated from: gold_standard_to_quantiles.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from datetime import datetime, timezone
import time

from wem.utils.logging import log
from wem.utils.columns import choose_col

# ---------------- helpers ----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Required columns (flexible names)
    dt_col  = choose_col(out, ["datetime", "time", "timestamp"])
    ws_col  = choose_col(out, ["ws_observed", "ws", "speed", "wind_speed"])
    sid_col = choose_col(out, ["site_id", "station_id", "id"])
    h_col   = choose_col(out, ["height", "height_m", "z"])
    lat_col = choose_col(out, ["lat", "latitude"])
    lon_col = choose_col(out, ["lon", "lng", "longitude"])
    name_col = choose_col(out, ["name", "station_name", "site_name"])
    elev_col = choose_col(out, ["elev_m", "elevation", "elevation_meters", "elev"])
    wst_col  = choose_col(out, ["windsite_type", "site_type", "type"])

    required = [dt_col, ws_col, sid_col, h_col, lat_col, lon_col]
    if any(c is None for c in required):
        missing = ["datetime","ws_observed","site_id","height","lat","lon"]
        raise ValueError(f"Missing required columns; need {missing}, got {list(df.columns)}")

    out = out.rename(columns={
        dt_col:  "datetime",
        ws_col:  "ws_observed",
        sid_col: "site_id",
        h_col:   "height",
        lat_col: "lat",
        lon_col: "lon",
        (elev_col or "elev_m"): "elev_m",
    })
    if wst_col and wst_col != "windsite_type":
        out = out.rename(columns={wst_col: "windsite_type"})

    # Ensure a 'name' column exists: "<site_id>_<height>"
    if (name_col is None) or ("name" not in out.columns):
        out["name"] = out["site_id"].astype(str) + "_" + out["height"].astype(str)
    else:
        out = out.rename(columns={name_col: "name"})

    # Types
    if not is_datetime64_any_dtype(out["datetime"]):
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
    else:
        if out["datetime"].dt.tz is None:
            out["datetime"] = out["datetime"].dt.tz_localize("UTC")
        else:
            out["datetime"] = out["datetime"].dt.tz_convert("UTC")

    out["ws_observed"] = pd.to_numeric(out["ws_observed"], errors="coerce")
    out["height"] = pd.to_numeric(out["height"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    if "elev_m" in out.columns:
        out["elev_m"] = pd.to_numeric(out["elev_m"], errors="coerce")

    # Basic cleaning
    keep = (
        out["site_id"].notna()
        & out["height"].notna()
        & out["lat"].notna()
        & out["lon"].notna()
        & out["datetime"].notna()
    )
    out = out.loc[keep].copy()

    # Non-negative wind speeds
    out["ws_observed"] = out["ws_observed"].where(out["ws_observed"] >= 0, np.nan)

    return out

def compute_group_quantiles(s: pd.Series) -> pd.Series:
    """Return q000..q100 for a group's ws_observed (NaNs OK)."""
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]

    keys = [f"q{p:03d}" for p in range(101)]
    if arr.size == 0:
        return pd.Series({k: np.nan for k in keys})

    pcts = np.arange(101, dtype=float)  # 0..100
    try:
        vals = np.nanpercentile(arr, pcts, method="linear")         # NumPy >=1.22
    except TypeError:
        vals = np.nanpercentile(arr, pcts, interpolation="linear")  # older NumPy
    return pd.Series({k: float(v) for k, v in zip(keys, vals)})

def mode_or_first(s: pd.Series) -> str:
    if s is None or s.empty:
        return ""
    m = s.mode(dropna=True)
    if m.empty:
        nonnull = s.dropna()
        return str(nonnull.iloc[0]) if not nonnull.empty else ""
    return str(m.iloc[0])

def month_balance_filter_table(df: pd.DataFrame,
                               start_year: int,
                               end_year: int,
                               min_months: int,
                               min_frac_of_median: float) -> pd.DataFrame:
    """
    Build a per-(site_id,height,year) table with monthly counts and keep/why flags.
    Does NOT modify the original df.
    """
    d = df.copy()
    d["year"] = d["datetime"].dt.year
    d["month"] = d["datetime"].dt.month
    d = d[(d["year"] >= start_year) & (d["year"] <= end_year)]

    # Monthly counts per site_id/height/year
    counts = (
        d.groupby(["site_id", "height", "year", "month"], sort=False)
         .size()
         .rename("count")
         .reset_index()
    )

    # Pivot months into columns 1..12
    piv = counts.pivot_table(index=["site_id", "height", "year"],
                             columns="month", values="count", fill_value=0)
    # Ensure all 12 months present as columns
    for m in range(1, 13):
        if m not in piv.columns:
            piv[m] = 0
    piv = piv.reindex(columns=list(range(1, 13)), fill_value=0)

    # Diagnostics
    nz = piv.replace(0, np.nan)
    median_nonzero = nz.median(axis=1, skipna=True).fillna(0.0)
    months_present = (piv > 0).sum(axis=1).astype(int)
    threshold = median_nonzero * float(min_frac_of_median)
    months_ge_threshold = piv.ge(threshold, axis=0).sum(axis=1).astype(int)

    # Keep rule
    kept = (months_present >= int(min_months)) & (months_ge_threshold >= int(min_months))

    # Windsite type (mode) for info
    wtype = (d.groupby(["site_id", "height", "year"])["windsite_type"]
               .agg(mode_or_first) if "windsite_type" in d.columns else pd.Series(dtype=str))
    wtype.name = "windsite_type"

    # Total samples
    total_count = piv.sum(axis=1).astype(int)

    out = pd.DataFrame({
        "months_present": months_present,
        "median_nonzero": median_nonzero,
        "threshold": threshold,
        "months_ge_threshold": months_ge_threshold,
        "total_count": total_count,
        "kept": kept,
    }, index=piv.index).reset_index()

    # Attach windsite_type if available
    if not wtype.empty:
        out = out.merge(wtype.reset_index(), on=["site_id", "height", "year"], how="left")

    # Attach month columns m01..m12
    month_cols = {m: f"m{m:02d}" for m in range(1, 13)}
    piv_named = piv.rename(columns=month_cols).reset_index()
    out = out.merge(piv_named, on=["site_id", "height", "year"], how="left")

    # Reason text for filtered rows
    reason = np.where(~kept, np.where(months_present < int(min_months),
                                      "insufficient_months_present",
                                      "insufficient_months_ge_threshold"),
                      "ok")
    out["reason"] = reason

    return out

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Gold-standard observations -> per-site quantiles (2007-2024, month-balance filtered; no rounding/dedup).")
    ap.add_argument("--pkl",  type=Path, default=Path("../data/gold_standard_timeseries.pkl"),
                    help="Pickle file containing the observations DataFrame.")
    ap.add_argument("--out",  type=Path, default=Path("gold_standard_quantiles.csv"),
                    help="Output CSV path.")
    ap.add_argument("--filtered-out-csv", type=Path, default=Path("gold_standard_filtered_out.csv"),
                    help="Where to write diagnostics for filtered-out site-years.")
    ap.add_argument("--start-year", type=int, default=2007)
    ap.add_argument("--end-year",   type=int, default=2024)
    ap.add_argument("--min-months", type=int, default=12,
                    help="Minimum months with data per kept year.")
    ap.add_argument("--min-frac-of-median", type=float, default=0.5,
                    help="A month is 'well represented' if count >= this * median(nonzero monthly counts); need at least min-months such months.")
    args = ap.parse_args()

    # Load
    log(f"[INFO] Loading dataframe: {args.pkl}")
    df = pd.read_pickle(args.pkl)
    log(f"[INFO] Loaded {len(df):,} rows with columns: {list(df.columns)}")

    # Normalize (no rounding or dedup)
    df = normalize_columns(df)
    log(f"[INFO] After basic cleaning (no rounding/dedup): {len(df):,} rows")

    # Build month-balance table & persist filtered-out diagnostics
    log(f"[INFO] Evaluating month balance per (site_id,height,year) for {args.start_year}-{args.end_year} ...")
    mb = month_balance_filter_table(
        df,
        start_year=args.start_year,
        end_year=args.end_year,
        min_months=args.min_months,
        min_frac_of_median=args.min_frac_of_median,
    )

    # Save filtered-out diagnostics
    filt_out = mb[~mb["kept"]].copy()
    args.filtered_out_csv.parent.mkdir(parents=True, exist_ok=True)
    filt_out.to_csv(args.filtered_out_csv, index=False)
    log(f"[INFO] Wrote filtered-out diagnostics: {len(filt_out)} site-year(s) -> {args.filtered_out_csv.resolve()}")

    # Keep only rows from KEPT site-years
    keep_keys = mb.loc[mb["kept"], ["site_id", "height", "year"]]
    if keep_keys.empty:
        log("[WARN] No site-years passed the month-balance filter; writing empty output with header.")
        cols_q = [f"q{p:03d}" for p in range(101)]
        empty = pd.DataFrame(columns=["station_id", "name", "lat", "lon", "elev_m",
                                      "dataset", "height_m", "interp", "agg", "years", "processed_utc"] + cols_q)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(args.out, index=False)
        return
    # Per-(site_id, height) list of kept years, e.g. "2008,2010,2012"
    years_map = (
        keep_keys.groupby(["site_id", "height"])["year"]
        .apply(lambda s: ",".join(str(int(y)) for y in sorted(s.unique())))
        .reset_index(name="years")
    )


    df["year"] = df["datetime"].dt.year
    key_df = keep_keys.copy()
    df_kept = df.merge(key_df, on=["site_id", "height", "year"], how="inner").drop(columns=["year"])
    log(f"[INFO] After filtering to kept site-years: {len(df_kept):,} rows")

    # Grouping key: (site_id, height)
    groups = df_kept.groupby(["site_id", "height"], sort=False)
    log(f"[INFO] Unique (site_id, height) groups after filter: {groups.ngroups}")

    # Quantiles per group
    log(f"[INFO] Computing quantiles q000..q100 per group (linear interpolation)...")
    qs = groups["ws_observed"].apply(compute_group_quantiles).unstack()

    # Metadata per group (based on kept data)
    log(f"[INFO] Aggregating metadata per group...")
    def summarize_group_meta(g: pd.DataFrame) -> dict:
        sid, height = g.name
        return {
            "site_id":  sid,
            "height":   float(height) if pd.notna(height) else np.nan,
            "name":     str(g["name"].iloc[0]) if "name" in g.columns and g["name"].notna().any() else f"{sid}_{str(height)}",
            "lat":      float(np.nanmedian(g["lat"].to_numpy(dtype="float64"))) if "lat" in g.columns else np.nan,
            "lon":      float(np.nanmedian(g["lon"].to_numpy(dtype="float64"))) if "lon" in g.columns else np.nan,
            "elev_m":   (float(np.nanmedian(g["elev_m"].to_numpy(dtype="float64")))
                         if "elev_m" in g.columns else np.nan),
            "t_min":    g["datetime"].min(),
            "t_max":    g["datetime"].max(),
            "n":        int(g.shape[0]),
        }

    meta = groups.apply(summarize_group_meta).apply(pd.Series)

    # Merge meta + quantiles
    out = meta.join(qs, how="left").reset_index(drop=True)
    out = out.merge(years_map, on=["site_id", "height"], how="left")
    out["years"] = out["years"].fillna("")

    # Final schema
    cols_q = [f"q{p:03d}" for p in range(101)]
    out = out.rename(columns={"site_id": "station_id", "height": "height_m"})
    out["dataset"] = "OBS"
    out["interp"] = "observed"
    out["agg"] = "native"

    out["processed_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "+00:00Z")

    out = out.reindex(columns=[
        "station_id", "name", "lat", "lon", "elev_m",
        "dataset", "height_m", "interp", "agg", "years", "processed_utc"
    ] + cols_q)

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    log(f"[INFO] Wrote {len(out):,} site-rows with quantiles to: {args.out.resolve()}")

if __name__ == "__main__":
    main()
