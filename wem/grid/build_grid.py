#!/usr/bin/env python3
"""
Build ERA5 grid_id,lat,lon from an ERA5 quantiles table (no pairwise de-dup).

- Input: ERA5 quantiles file (CSV or Parquet) that includes latitude/longitude columns.
- Output: CSV with exactly: grid_id, lat, lon
- grid_id: "LLLPPP" where
    LLL = zero-padded latitude index  (0..Nlat-1), assigned in DESCENDING latitude (north->south)
    PPP = zero-padded longitude index (0..Nlon-1), assigned in ASCENDING longitude (west->east)

Notes:
- We do NOT drop rows; we assume each (lat,lon) row is unique already.
- We round lat/lon to a fixed number of decimals to stabilize the mapping against tiny float jitter,
  but we write out the original (unrounded) lat/lon in the output.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from wem.utils.columns import choose_col


def load_coords(path: Path) -> pd.DataFrame:
    """Load only lat/lon columns from CSV or Parquet; keep original precision."""
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        # Try to read only plausible lat/lon columns if CSV to save memory
        try:
            df = pd.read_csv(path, usecols=lambda c: c.lower() in {"lat","latitude","lon","longitude"})
        except Exception:
            df = pd.read_csv(path)

    latc = choose_col(df, ["lat","latitude"])
    lonc = choose_col(df, ["lon","longitude"])
    if not latc or not lonc:
        raise ValueError("Input must include latitude/longitude columns (lat/latitude, lon/longitude).")

    out = df[[latc, lonc]].rename(columns={latc: "lat", lonc: "lon"}).copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat","lon"]).reset_index(drop=True)
    return out


def assign_indices(points: pd.DataFrame, round_decimals: int = 6) -> pd.DataFrame:
    """
    Assign (lat_idx, lon_idx) to each row, without dropping any rows.
    - Latitude indices: descending lat (north->south).
    - Longitude indices: ascending lon (west->east).
    """
    pts = points.copy()

    # Stabilize mapping keys with rounding (ERA5 grid is regular; this avoids tiny float jitter)
    lat_r = pts["lat"].round(round_decimals).to_numpy(dtype="float64")
    lon_r = pts["lon"].round(round_decimals).to_numpy(dtype="float64")

    # Build axes from unique values in each column
    lat_axis = np.unique(lat_r)[::-1]     # descending
    lon_axis = np.unique(lon_r)           # ascending

    # Maps value -> index
    lat_to_idx = {v: i for i, v in enumerate(lat_axis)}
    lon_to_idx = {v: j for j, v in enumerate(lon_axis)}

    # Vectorized index assignment via mapping (use pandas map for simplicity)
    pts["lat_idx"] = pd.Series(lat_r).map(lat_to_idx).to_numpy(dtype=int)
    pts["lon_idx"] = pd.Series(lon_r).map(lon_to_idx).to_numpy(dtype=int)

    # Build 6-digit grid_id = LLLPPP
    L = pts["lat_idx"].to_numpy()
    P = pts["lon_idx"].to_numpy()
    grid_id = np.char.add(np.char.zfill(L.astype(str), 3), np.char.zfill(P.astype(str), 3))

    result = pd.DataFrame({
        "grid_id": grid_id,
        "lat": pts["lat"].to_numpy(dtype="float64"),
        "lon": pts["lon"].to_numpy(dtype="float64"),
    })

    # Optional: stable ordering (not required)
    result = result.sort_values(["grid_id"]).reset_index(drop=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="Create ERA5 grid_id,lat,lon from an ERA5 quantiles table (no de-dup).")
    ap.add_argument("--in",  dest="inp",  type=Path, required=True, help="Input ERA5 quantiles file (CSV or Parquet).")
    ap.add_argument("--out", dest="outp", type=Path, required=True, help="Output CSV with columns: grid_id,lat,lon.")
    ap.add_argument("--round-decimals", type=int, default=6,
                    help="Rounding for index mapping (default: 6). Increase if needed; does not affect output lat/lon.")
    ap.add_argument("--assert-shape", type=str, default=None,
                    help='Optional "NlatxNlon" assertion, e.g., "113x375".')
    args = ap.parse_args()

    df = load_coords(args.inp)
    out = assign_indices(df, round_decimals=args.round_decimals)

    # Optional sanity: enforce expected grid shape, e.g., 113x375
    if args.assert_shape:
        try:
            nlat_s, nlon_s = args.assert_shape.lower().split("x")
            nlat_e = int(nlat_s); nlon_e = int(nlon_s)
            nlat = out["grid_id"].str[:3].astype(int).max() + 1
            nlon = out["grid_id"].str[3:].astype(int).max() + 1
            if (nlat, nlon) != (nlat_e, nlon_e):
                raise AssertionError(f"Detected shape {nlat}x{nlon} != expected {nlat_e}x{nlon_e}")
        except Exception as e:
            raise SystemExit(f"--assert-shape failed: {e}")

    out.to_csv(args.outp, index=False)


if __name__ == "__main__":
    main()
