"""
Merge per-tile grid extraction outputs into one CSV per height.

Consolidates the logic from the three standalone merge scripts
(merge_wtk.py, merge_hrrr.py, merge_wtk_led.py) into a single
CLI command parameterized by --prefix.

Input:
  - Directory of tile files (CSV or Parquet) produced by
    wem-grid-wtk / wem-grid-hrrr / wem-grid-wtkled
  - Each tile contains: grid_id, lat, lon, height_m, q000..q100

Output:
  - One CSV per height value, named {prefix}_quantiles_{height}m.csv
  - Columns: grid_id, lat, lon, height_m, q000..q100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wem.constants import QCOLS
from wem.utils.logging import log

KEEP = ["grid_id", "lat", "lon", "height_m"] + QCOLS


def list_input_files(in_dir: Path) -> list[Path]:
    """Find tile CSV/Parquet/PQ files in a directory."""
    files: list[Path] = []
    for ext in ("*.parquet", "*.pq", "*.csv"):
        files += sorted(in_dir.glob(ext))
    return files


def read_tile(fp: Path) -> pd.DataFrame:
    """Read a single tile file, keeping only KEEP columns, normalizing types."""
    if fp.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(fp)
    else:
        try:
            keep_set = set(KEEP)
            df = pd.read_csv(fp, usecols=lambda c: c in keep_set)
        except Exception:
            df = pd.read_csv(fp)

    missing = [c for c in ("grid_id", "lat", "lon", "height_m") if c not in df.columns]
    if missing:
        raise ValueError(f"{fp.name} missing required columns: {missing}")

    missing_qcols = [qc for qc in QCOLS if qc not in df.columns]
    if missing_qcols:
        log(f"[WARN] {fp.name} missing {len(missing_qcols)} quantile columns: "
            f"{missing_qcols[0]}..{missing_qcols[-1]} — filling with NaN")
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing_qcols)],
            axis=1,
        )

    df = df[KEEP].copy()
    df["grid_id"] = df["grid_id"].astype("string")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    return df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge per-tile grid extraction outputs into one CSV per height."
    )
    ap.add_argument(
        "--in-dir", type=Path, required=True,
        help="Directory containing tile CSV/Parquet files.",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory for per-height output CSVs.",
    )
    ap.add_argument(
        "--prefix", type=str, required=True,
        help="Output filename prefix (e.g. wtk, hrrr, wtk_led).",
    )
    ap.add_argument(
        "--heights", type=str, default=None,
        help="Comma-separated height filter (e.g. 30,40,60,80,100).",
    )
    ap.add_argument(
        "--dedupe", action="store_true",
        help="Drop duplicate (grid_id, height_m) rows per output.",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Remove existing output CSVs before writing.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = list_input_files(args.in_dir)
    if not files:
        raise SystemExit(f"No input files found in {args.in_dir}")

    wanted_heights: set[int] | None = None
    if args.heights:
        wanted_heights = {
            int(round(float(h))) for h in args.heights.split(",") if h.strip()
        }

    if args.overwrite:
        for fp in args.out_dir.glob(f"{args.prefix}_quantiles_*m.csv"):
            try:
                fp.unlink()
            except Exception:
                pass

    log(f"Found {len(files)} input file(s). Starting merge…")
    total_rows = 0
    per_h_rows: dict[int, int] = {}

    for i, fp in enumerate(files, 1):
        log(f"[{i}/{len(files)}] Reading {fp.name} …")
        df = read_tile(fp)

        z_int = df["height_m"].round(0).astype(int)
        if wanted_heights is not None:
            mask = z_int.isin(wanted_heights)
            df, z_int = df[mask], z_int[mask]

        if df.empty:
            continue

        for z_val, g in df.groupby(z_int):
            out_fp = args.out_dir / f"{args.prefix}_quantiles_{z_val}m.csv"

            out_block = g[KEEP].copy()

            if args.dedupe:
                before = len(out_block)
                out_block = out_block.drop_duplicates(subset=["grid_id", "height_m"])
                if len(out_block) < before:
                    log(f"  - Deduped height {z_val}m: {before} -> {len(out_block)}")

            header = not out_fp.exists()
            out_block.to_csv(out_fp, mode="a", header=header, index=False)

            n = len(out_block)
            per_h_rows[z_val] = per_h_rows.get(z_val, 0) + n
            total_rows += n

    log("Merge complete.")
    for z in sorted(per_h_rows):
        log(f"  Height {z} m → {per_h_rows[z]:,} row(s)")
    log(f"Total rows written: {total_rows:,}")


if __name__ == "__main__":
    main()
