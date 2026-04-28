#!/usr/bin/env python3
"""
Process raw ISD wind CSVs into compact quantile files with ASOS de-quantization.

For each station file in wind_data_by_station/:
1) Extract wind speed (m/s) from WND.
2) Detect nominal samples/day; drop years with <95% completeness.
3) De-quantize ASOS-reported speeds using the 'calm<=2 kt, else ceil' model.
4) Build a CDF with 101 points (0..100%).
5) Write processed_data/<station_id>_quantiles.csv with a JSON header.

Restart-safe: existing processed files are skipped.

Migrated from: process_isd_to_quantiles.py
"""

from __future__ import annotations

import argparse
import os, sys, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from wem.utils.logging import log

# De-quantization settings (your model)
CALM_THRESHOLD_KT = 2.0        # calm clamp: true <= 2 kt -> reported 0 kt
KNOT_TO_MS = 0.514444

# ── UTILITIES ────────────────────────────────────────────────

def days_in_year(year: int) -> int:
    return 366 if (year % 400 == 0 or (year % 4 == 0 and year % 100)) else 365

# ── METADATA ─────────────────────────────────────────────────
def load_metadata(meta_csv: Path) -> pd.DataFrame:
    if not meta_csv.exists():
        raise FileNotFoundError("Run the downloader first -- metadata file missing.")
    meta = pd.read_csv(meta_csv, dtype=str)
    return meta.set_index("station_id")

# ── WIND SPEED PARSING ──────────────────────────────────────
def parse_speed_ms(wnd_series: pd.Series) -> pd.Series:
    """Return wind speed in m/s from ISD WND (tenths of m/s); 9999->NaN."""
    def extract(s: str) -> float | np.nan:
        try:
            parts = s.split(",")
            if len(parts) < 4:
                return np.nan
            tenths = int(parts[3])
            return np.nan if tenths == 9999 else tenths / 10.0
        except Exception:
            return np.nan
    return wnd_series.astype(str).map(extract, na_action="ignore")

# ── COMPLETENESS CHECKS ─────────────────────────────────────
def expected_per_day(df: pd.DataFrame) -> int:
    counts = df.groupby(df["DATE"].dt.date).size()
    return int(counts.mode().iloc[0]) if not counts.empty else 0

def years_with_enough_data(df: pd.DataFrame, s_per_day: int) -> List[int]:
    df = df.copy()
    df["YEAR"] = df["DATE"].dt.year
    good: List[int] = []
    for y, n in df.groupby("YEAR").size().items():
        expect = s_per_day * days_in_year(int(y))
        if expect and n >= 0.95 * expect:
            good.append(int(y))
    return good

# ── DE-QUANTIZATION (ceil model) ────────────────────────────
def dequantize_ceil(speed_ms: pd.Series) -> pd.Series:
    """
    Implement:
      - if reported 0 kt -> sample U(0, 2] kt
      - if reported k kt (k >= 3) -> sample U(k-1, k] kt
    Where the reported knot 'k' is recovered from the stored m/s via k = round(ms / 0.514444).
    """
    s = speed_ms.copy()
    out = s.to_numpy(dtype=float)

    finite = np.isfinite(out)
    if not finite.any():
        return s

    # recover the reported integer knots from stored m/s (which is ~round(k*0.514444, 1))
    knots_est = np.rint(out[finite] / KNOT_TO_MS).astype(int)

    # calm if stored m/s == 0.0
    calm_mask = (out[finite] == 0.0)

    lo = knots_est.astype(float) - 1.0
    hi = knots_est.astype(float)       # U(k-1, k] for k>=3

    # for calm, use U(0, 2] kt
    lo[calm_mask] = 0.0
    hi[calm_mask] = CALM_THRESHOLD_KT

    # ensure bins make sense (guard pathological inputs)
    lo = np.where(hi <= 2.0, 0.0, lo)  # if somehow k<3, keep nonnegative
    u = np.random.random(size=lo.shape[0])
    sampled_knots = lo + u * (hi - lo)
    out[finite] = sampled_knots * KNOT_TO_MS
    return pd.Series(out, index=speed_ms.index, dtype=float)

# ── QUANTILES ───────────────────────────────────────────────
def make_quantiles(speed: pd.Series) -> pd.DataFrame:
    qs = np.linspace(0, 1, 101)
    vals = speed.quantile(qs, interpolation="linear").values
    return pd.DataFrame({"quantile": (qs * 100).round(0).astype(int),
                         "wind_speed_m_s": vals})

# ── WRITER ──────────────────────────────────────────────────
def write_processed(out_path: Path, header: dict, q_df: pd.DataFrame) -> None:
    with out_path.open("w", newline="") as fh:
        fh.write("# " + json.dumps(header, separators=(",", ":")) + "\n")
        q_df.to_csv(fh, index=False)

# ── CORE PROCESSOR ──────────────────────────────────────────
def process_one(raw_path: Path, proc_dir: Path, meta: pd.DataFrame,
                random_seed: Optional[int]) -> bool:
    station_id = raw_path.stem.split("_")[0]
    out_path   = proc_dir / f"{station_id}_quantiles.csv"
    if out_path.exists():
        log(f"{station_id} already processed")
        return True

    try:
        df = pd.read_csv(raw_path, parse_dates=["DATE"], usecols=["DATE", "WND"])
    except Exception as e:
        log(f"{station_id} load error: {e}")
        return False

    df["speed_ms_raw"] = parse_speed_ms(df["WND"]).astype(float)
    df = df.dropna(subset=["speed_ms_raw"])
    if df.empty:
        log(f"{station_id} no valid speeds")
        return False

    s_per_day = expected_per_day(df)
    good_years = years_with_enough_data(df, s_per_day)
    if not good_years:
        log(f"{station_id} no year meets 95 % completeness")
        return False

    df = df[df["DATE"].dt.year.isin(good_years)].copy()
    if df.empty:
        log(f"{station_id} no rows after year filter")
        return False

    # de-quantize using the ceil model
    df["speed_ms_adj"] = dequantize_ceil(df["speed_ms_raw"])

    # build quantiles from adjusted speeds
    q_df = make_quantiles(df["speed_ms_adj"])

    meta_row = meta.loc[station_id].to_dict() if station_id in meta.index else {}
    header = {
        **meta_row,
        "station_id": station_id,
        "samples_per_day": int(s_per_day),
        "years_used": ",".join(map(str, sorted(good_years))),
        "n_points": int(df.shape[0]),
        "dequantization": {
            "model": "ceil",
            "calm_threshold_knots": CALM_THRESHOLD_KT,
            "bin_for_k>=3": "(k-1, k]",
            "random_seed": random_seed
        }
    }

    write_processed(out_path, header, q_df)
    log(f"{station_id} processed ({len(good_years)} yrs, {df.shape[0]} pts)")
    return True

# ── MAIN ────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Process raw ISD wind CSVs into quantile files with ASOS de-quantization.",
    )
    ap.add_argument("--raw-dir", type=Path, default=Path("wind_data_by_station"),
                    help="Directory containing raw per-station wind CSVs.")
    ap.add_argument("--proc-dir", type=Path, default=Path("processed_data"),
                    help="Output directory for quantile files.")
    ap.add_argument("--meta-csv", type=Path,
                    default=Path("us_wind_station_metadata_2007_2024.csv"),
                    help="Station metadata CSV (from the downloader).")
    ap.add_argument("--max-workers", type=int, default=16,
                    help="Number of parallel processing threads.")
    ap.add_argument("--random-seed", type=int, default=12345,
                    help="Random seed for de-quantization sampling (use -1 for no seed).")
    args = ap.parse_args()

    args.proc_dir.mkdir(exist_ok=True, parents=True)

    random_seed: Optional[int] = args.random_seed if args.random_seed >= 0 else None
    if random_seed is not None:
        np.random.seed(random_seed)

    meta = load_metadata(args.meta_csv)

    raw_files = list(args.raw_dir.glob("*_wind_2007_2024.csv"))
    if not raw_files:
        log("No raw CSVs found. Run the downloader first.")
        return

    log(f"{len(raw_files)} raw files detected.")
    with ThreadPoolExecutor(max_workers=args.max_workers) as tp:
        futs = {
            tp.submit(process_one, p, args.proc_dir, meta, random_seed): p.stem
            for p in raw_files
        }
        for fut in as_completed(futs):
            fut.result()

    log("All done. Processed files in " + str(args.proc_dir))

if __name__ == "__main__":
    main()
