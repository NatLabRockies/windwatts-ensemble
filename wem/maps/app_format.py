#!/usr/bin/env python3
"""
Convert ML quantiles (site_quantiles_predicted_gwa.csv) into the app's per-location files
with optional interpolation to 33 probabilities. By default, outputs full 101 quantiles.

Adds end-of-run stats on adjacent-equal OUTPUT values:
  - sites_with_adjacent_equals
  - total_adjacent_equal_pairs

Inputs (CSV):
  - columns: index, latitude, longitude, height_m, q000..q100  (101 quantiles)

Outputs:
  - one file per location: {index}.csv.gz
    columns:
      probability, windspeed_30m, windspeed_40m, windspeed_50m,
      windspeed_60m, windspeed_80m, windspeed_100m
    * probability has 101 rows by default (0..1 in steps of 0.01)
      or 33 rows if --to-33 / --out-quantiles 33 is used.
    * index is ALWAYS a 6-character zero-padded string (e.g., '000123')

Optional:
  - if --era5-index is given, we filter locations to those IDs and write a filtered
    location_index.csv.gz (index, latitude, longitude)
  - if no --era5-index but --make-index is provided, we generate location_index.csv.gz
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional

import gzip
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wem.utils.logging import log

# ----- constants -----
TARGET_HEIGHTS = [30, 40, 50, 60, 80, 100]
QCOLS_101 = [f"q{p:03d}" for p in range(101)]
PIN_101   = np.linspace(0.0, 1.0, 101, dtype=np.float64)  # q000..q100
POUT_33   = np.linspace(0.0, 1.0, 33,  dtype=np.float64)  # 0..1 step=1/32

# ----- utils -----
def normalize_index_series(s: pd.Series) -> pd.Series:
    """
    Force every value to a 6-character, zero-padded string.
    Accepts integers, '123', '00123', or '123.0'. Raises on >6 digits or non-numeric.
    """
    def _one(x) -> str:
        if pd.isna(x):
            raise ValueError("Missing index value.")
        t = str(x).strip()
        # if like '123.0' or floaty, coerce to int
        try:
            if any(c in t for c in (".", "e", "E", "+")):
                n = int(float(t))
                t = str(n)
        except Exception:
            pass
        if t.isdigit():
            if len(t) > 6:
                raise ValueError(f"Index '{t}' has more than 6 digits.")
            return t.zfill(6)
        # last chance: numeric coercion
        try:
            n = int(float(t))
            t2 = str(n)
            if len(t2) > 6:
                raise ValueError(f"Index '{t2}' has more than 6 digits.")
            return t2.zfill(6)
        except Exception:
            raise ValueError(f"Index '{x}' is not numeric.")
    return s.apply(_one)

def enforce_monotonic(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64, copy=True)
    if np.any(~np.isfinite(q)):
        q = pd.Series(q).interpolate("linear", limit_direction="both").to_numpy(dtype=float)
    q = np.nan_to_num(q, nan=0.0)
    return np.maximum.accumulate(q)

def to_output_profile(q101: np.ndarray, out_quantiles: int) -> np.ndarray:
    """Return the OUTPUT profile (length 101 by default, or 33 if requested)."""
    q101m = enforce_monotonic(q101)
    if out_quantiles == 33:
        return np.interp(POUT_33, PIN_101, q101m).astype(np.float64)
    # default: keep full 101
    return q101m

def power_alpha(v1: np.ndarray, v2: np.ndarray, h1: float, h2: float) -> np.ndarray:
    v1c = np.clip(v1, 1e-6, None)
    v2c = np.clip(v2, 1e-6, None)
    denom = np.log(h2 / h1)
    if not np.isfinite(denom) or denom == 0.0:
        denom = 1.0
    a = np.log(v2c / v1c) / denom
    return np.clip(a, -2.0, 2.0)

def vertical_fill_one(
    want_h: int,
    have: Dict[int, np.ndarray],
    out_quantiles: int,
    fallback_alpha: float = 1/7
) -> Optional[np.ndarray]:
    """Power-law vertical interpolation at the OUTPUT quantile grid."""
    if want_h in have:
        return have[want_h]
    hs = sorted(have.keys())
    if not hs:
        return None
    h_lo = max([h for h in hs if h < want_h], default=None)
    h_hi = min([h for h in hs if h > want_h], default=None)
    if h_lo is not None and h_hi is not None:
        v_lo, v_hi = have[h_lo], have[h_hi]
        a = power_alpha(v_lo, v_hi, float(h_lo), float(h_hi))
        return (v_lo * ((want_h / float(h_lo)) ** a)).astype(np.float64)
    # single neighbor -> fallback exponent
    h_near = min(hs, key=lambda h: abs(h - want_h))
    v_near = have[h_near]
    return (v_near * ((want_h / float(h_near)) ** float(fallback_alpha))).astype(np.float64)

def write_loc_file(out_path: Path, ws_by_h: Dict[int, np.ndarray], out_quantiles: int) -> None:
    # pick the correct probability vector
    prob = PIN_101 if out_quantiles == 101 else POUT_33
    cols = ["probability"] + [f"windspeed_{h}m" for h in TARGET_HEIGHTS]
    mat = np.zeros((len(prob), len(cols)), dtype=np.float64)
    mat[:, 0] = prob
    for j, h in enumerate(TARGET_HEIGHTS, start=1):
        mat[:, j] = ws_by_h[h]
    out_df = pd.DataFrame(mat, columns=cols)
    with gzip.open(out_path, "wt", newline="") as f:
        out_df.to_csv(f, index=False)

def count_adjacent_equals(arr: np.ndarray, tol: float = 0.0) -> int:
    """Count pairs where arr[i] and arr[i+1] are equal (or within tol)."""
    a = np.asarray(arr, dtype=np.float64)
    diffs = np.abs(a[1:] - a[:-1])
    if tol <= 0.0:
        return int(np.sum(diffs == 0.0))
    return int(np.sum(diffs <= tol))

# ----- main -----
def main():
    ap = argparse.ArgumentParser(
        description="Build {index}.csv.gz files (6-char names) from ML predictions; optional interpolation to 33 probabilities."
    )
    ap.add_argument("--in", dest="infile", type=Path, required=True,
                    help="site_quantiles_predicted_gwa.csv (index,latitude,longitude,height_m,q000..q100)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for {index}.csv.gz")
    ap.add_argument("--era5-index", type=Path, default=None,
                    help="Optional ERA5 index CSV (columns: index, latitude, longitude)")
    ap.add_argument("--make-index", action="store_true",
                    help="If no --era5-index, also write location_index.csv.gz built from ML file.")
    ap.add_argument("--skip-missing", action="store_true", default=True,
                    help="Skip locations with no usable heights instead of erroring (default: True).")
    ap.add_argument("--strict", action="store_true",
                    help="Fail immediately if any location has no usable heights.")
    ap.add_argument("--equal-tol", type=float, default=0.0,
                    help="Tolerance for treating adjacent outputs as equal (default 0 = exact equality).")
    # New controls for output quantiles
    ap.add_argument("--to-33", action="store_true",
                    help="If set, interpolate to 33 probabilities (0..1 step 1/32). Default: keep 101.")
    ap.add_argument("--out-quantiles", type=int, choices=[101, 33], default=None,
                    help="Explicitly choose 101 or 33 (overrides --to-33 if provided).")
    args = ap.parse_args()

    out_quantiles = (33 if args.to_33 else 101)
    if args.out_quantiles is not None:
        out_quantiles = args.out_quantiles

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ML predictions ---
    log(f"Loading ML predictions: {args.infile}")
    usecols = ["index", "latitude", "longitude", "height_m"] + QCOLS_101
    df = pd.read_csv(args.infile, dtype={"index": str}, low_memory=False, usecols=lambda c: (c in usecols))
    # normalize indices to 6-char
    df["index"] = normalize_index_series(df["index"])
    # clean coords / heights
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["height_m"]  = pd.to_numeric(df["height_m"], errors="coerce")
    df = df.dropna(subset=["index", "latitude", "longitude", "height_m"]).copy()
    df["height_m"] = df["height_m"].round().astype(int)

    # keep only target heights (ensures app compatibility)
    df = df[df["height_m"].isin(TARGET_HEIGHTS)].reset_index(drop=True)

    # --- optional ERA5 index alignment ---
    if args.era5_index is not None:
        log(f"Reading ERA5 location index: {args.era5_index}")
        idx = pd.read_csv(args.era5_index, dtype={"index": str})
        rename_map = {}
        for a, b in [("Latitude","latitude"), ("LATITUDE","latitude"), ("lat","latitude"),
                     ("Longitude","longitude"), ("LONGITUDE","longitude"), ("lon","longitude")]:
            if a in idx.columns:
                rename_map[a] = b
        idx = idx.rename(columns=rename_map)
        if "latitude" not in idx.columns or "longitude" not in idx.columns:
            raise SystemExit("ERA5 index must have columns 'index, latitude, longitude'.")
        idx["index"] = normalize_index_series(idx["index"])
        idx["latitude"]  = pd.to_numeric(idx["latitude"], errors="coerce")
        idx["longitude"] = pd.to_numeric(idx["longitude"], errors="coerce")
        idx = idx.dropna(subset=["index","latitude","longitude"])
        keep = set(idx["index"].unique())
        before = df["index"].nunique()
        df = df[df["index"].isin(keep)]
        after = df["index"].nunique()
        log(f"Restricting to ERA5 index IDs: kept {after:,}/{before:,} ML locations.")
        # write filtered index to output dir (gzip)
        out_loc = args.out_dir / "location_index.csv.gz"
        idx_out = idx.loc[idx["index"].isin(df["index"].unique()),
                          ["index","latitude","longitude"]].drop_duplicates()
        log(f"Writing location index -> {out_loc.name} ({len(idx_out):,} rows)")
        idx_out.to_csv(out_loc, index=False, compression="gzip")
    else:
        if args.make_index:
            out_loc = args.out_dir / "location_index.csv.gz"
            idx_out = df.groupby("index", as_index=False).agg(
                latitude=("latitude","first"), longitude=("longitude","first")
            )
            log(f"Writing generated location index -> {out_loc.name} ({len(idx_out):,} rows)")
            idx_out.to_csv(out_loc, index=False, compression="gzip")

    # --- verify quantile columns ---
    qcols_exist = [c for c in QCOLS_101 if c in df.columns]
    if len(qcols_exist) != 101:
        missing = sorted(set(QCOLS_101) - set(qcols_exist))[:5]
        raise SystemExit(f"Input missing expected quantile columns (e.g., {missing})")

    # --- build per-location OUTPUT quantile arrays (length 101 by default, 33 if requested) ---
    log(f"Building per-location output profiles ... (out_quantiles={out_quantiles})")
    records: Dict[str, Dict[int, np.ndarray]] = {}
    for (idx6, h), g in df.groupby(["index","height_m"], sort=False):
        row = g.iloc[0]
        q101 = row[qcols_exist].to_numpy(dtype=np.float64, copy=False)
        records.setdefault(idx6, {})[h] = to_output_profile(q101, out_quantiles)

    # --- write files + gather adjacent-equal stats on OUTPUT arrays ---
    n_written = n_skipped = 0
    total_adjacent_equal_pairs = 0
    sites_with_adjacent_equals = 0
    log("Writing {index}.csv.gz files ...")
    for idx6, have in tqdm(records.items(), total=len(records), unit="loc"):
        ws_by_h: Dict[int, np.ndarray] = {}
        for h in TARGET_HEIGHTS:
            v = vertical_fill_one(h, have, out_quantiles)
            if v is None:
                ws_by_h = {}
                break
            ws_by_h[h] = v

        if not ws_by_h:
            if args.strict:
                raise SystemExit(f"Location {idx6}: no usable heights to build output.")
            n_skipped += 1
            continue

        # Count adjacent-equal pairs across all heights for this site (OUTPUT arrays)
        site_pairs = 0
        for h in TARGET_HEIGHTS:
            site_pairs += count_adjacent_equals(ws_by_h[h], tol=args.equal_tol)
        if site_pairs > 0:
            sites_with_adjacent_equals += 1
            total_adjacent_equal_pairs += site_pairs

        out_fp = args.out_dir / f"{idx6}.csv.gz"   # ALWAYS 6-char name
        write_loc_file(out_fp, ws_by_h, out_quantiles)
        n_written += 1

    log(f"Done. Wrote {n_written:,} files; skipped {n_skipped:,}.")
    log(f"[ADJACENT-EQUAL OUTPUTS] sites_with_adjacent_equals={sites_with_adjacent_equals:,} | "
        f"total_adjacent_equal_pairs={total_adjacent_equal_pairs:,} | equal_tol={args.equal_tol:g} | "
        f"out_quantiles={out_quantiles}")

if __name__ == "__main__":
    main()
