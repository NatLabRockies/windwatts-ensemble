#!/usr/bin/env python3
"""
Aggregate station quantile files into one matrix CSV.

Input files (from your previous step):
  processed_data/<station_id>_quantiles.csv
    - line 1: "# { ... JSON header with metadata ... }"
    - body : columns: quantile (0..100), wind_speed_m_s

Output:
  all_sites_quantiles_2007_2024.csv
    - one row per site with >= 5 years (based on header.years_used)
    - metadata columns from the header (union across sites)
    - quantile columns q000..q100 (wind_speed_m_s)

Migrated from: aggregate_quantiles_to_matrix.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# You can prioritize / order some metadata keys up front (if present),
# the rest will follow in alphabetical order for reproducibility.
PREFERRED_META_ORDER = [
    "station_id", "USAF", "WBAN", "STATION NAME", "NAME", "CTRY", "STATE",
    "LAT", "LON", "ELEV(M)", "samples_per_day", "years_used", "n_points"
]

def read_header_and_table(path: Path) -> tuple[Dict, pd.DataFrame]:
    """Read first line JSON header (after '# ') and the quantile table."""
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
        if not first.startswith("#"):
            raise ValueError(f"{path.name}: Missing JSON header line starting with '# '")
        header = json.loads(first.lstrip("# ").strip())
    # now read the rest of the file as CSV
    df = pd.read_csv(path, comment="#")
    # Ensure expected columns
    if not {"quantile", "wind_speed_m_s"}.issubset(df.columns):
        raise ValueError(f"{path.name}: Missing required columns 'quantile' and 'wind_speed_m_s'")
    return header, df[["quantile", "wind_speed_m_s"]]

def years_from_header(header: Dict) -> List[int]:
    """Parse years_used from header into a list of ints; returns [] if missing/invalid."""
    years_str = header.get("years_used", "")
    years = []
    for tok in str(years_str).split(","):
        tok = tok.strip()
        if tok.isdigit():
            years.append(int(tok))
    return years

def quantile_columns() -> List[str]:
    return [f"q{q:03d}" for q in range(101)]

def build_row(header: Dict, qtab: pd.DataFrame) -> Dict:
    """
    Build a single output row:
      - metadata (from header)
      - q000..q100 mapped from quantile table
    """
    row: Dict = {}

    # Copy all metadata keys
    for k, v in header.items():
        # flatten nested dicts (e.g., 'dequantization') into JSON strings
        if isinstance(v, (dict, list)):
            row[k] = json.dumps(v, separators=(",", ":"))
        else:
            row[k] = v

    # Quantiles: ensure exactly 0..100 present; if missing, reindex and interpolate
    qtab = qtab.copy()
    qtab = qtab.dropna(subset=["quantile", "wind_speed_m_s"])
    qtab["quantile"] = qtab["quantile"].astype(int)

    # pivot to a Series indexed by quantile
    s = qtab.set_index("quantile")["wind_speed_m_s"].sort_index()

    # reindex 0..100; fill by interpolation (linear) if a few are missing
    full_index = pd.Index(range(101), name="quantile")
    s = s.reindex(full_index)
    if s.isna().any():
        s = s.interpolate(limit_direction="both")

    # attach as q000..q100
    for q in range(101):
        row[f"q{q:03d}"] = float(s.loc[q]) if pd.notna(s.loc[q]) else None

    return row

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate per-station quantile files into one matrix CSV.",
    )
    ap.add_argument("--proc-dir", type=Path, default=Path("processed_data"),
                    help="Directory containing per-station quantile CSVs.")
    ap.add_argument("--out-csv", type=Path,
                    default=Path("all_sites_quantiles_2007_2024.csv"),
                    help="Output aggregated matrix CSV.")
    ap.add_argument("--min-years", type=int, default=5,
                    help="Minimum number of years required to include a site.")
    args = ap.parse_args()

    files = sorted(args.proc_dir.glob("*_quantiles.csv"))
    if not files:
        print(f"No processed quantile files found in {args.proc_dir}")
        return

    rows: List[Dict] = []
    meta_keys_union: Set[str] = set()

    for path in files:
        try:
            header, qtab = read_header_and_table(path)
            yrs = years_from_header(header)
            if len(set(yrs)) < args.min_years:
                # Skip sites with fewer than min_years of data
                continue
            row = build_row(header, qtab)
            rows.append(row)
            meta_keys_union.update(k for k in row.keys() if not k.startswith("q"))
        except Exception as e:
            # be tolerant; skip problematic files
            print(f"Skipping {path.name}: {e}")

    if not rows:
        print(f"No eligible sites (>={args.min_years} years) found.")
        return

    # Assemble column order: preferred metadata (that actually exist) + other metadata + quantiles
    preferred = [k for k in PREFERRED_META_ORDER if k in meta_keys_union]
    remaining_meta = sorted(meta_keys_union - set(preferred))
    cols = preferred + remaining_meta + quantile_columns()

    df = pd.DataFrame(rows)
    # Make sure all expected columns exist (missing metas become NaN)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  ({len(df)} sites, {len(cols)} columns)")

if __name__ == "__main__":
    main()
