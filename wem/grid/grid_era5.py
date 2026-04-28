#!/usr/bin/env python3
"""
ERA5 grid-wide wind-speed quantiles at multiple heights (2007-2024), using local GRIBs.

For each grid cell:
  - compute alpha from speeds at 10 m and 100 m (fallback alpha = 1/7 where invalid)
  - interpolate speed to 30, 40, 60, 80 m (power law), use native 100 m
  - compute quantiles q0..q100 over time
  - write one CSV per height with columns: lat, lon, q000..q100

Inputs (must exist)
------
<era5-dir>/conus-YYYY-hourly.grib
  (must contain shortName 10u,10v,100u,100v)

Outputs
-------
era5_quantiles_grid_{height_m}m_2007_2024.csv  (one file per height)
"""

from __future__ import annotations
import argparse
import os, time, math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import xarray as xr

from wem.utils.logging import log

# ---- CONFIG DEFAULTS ----
DEFAULT_FILENAME_PATTERN = "conus-{year}-hourly.grib"
DEFAULT_START_YEAR = 2007
DEFAULT_END_YEAR   = 2024
DEFAULT_OUT_HEIGHTS = "50"
DEFAULT_OUT_CSV_PATTERN = "era5_quantiles_grid_{height_m:.0f}m_2007_2024_NEW.csv"

# Neutral power-law fallback exponent
ALPHA_FALLBACK = 1.0 / 7.0

# ---- ERA5 OPEN/MERGE ----
def grib_paths(era5_dir: Path, filename_pattern: str, years: List[int]) -> List[Path]:
    return [era5_dir / filename_pattern.format(year=y) for y in years]

def _rename_uv_vars(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "10u" in ds:  ren["10u"] = "u10"
    if "10v" in ds:  ren["10v"] = "v10"
    if "100u" in ds: ren["100u"] = "u100"
    if "100v" in ds: ren["100v"] = "v100"
    if ren:
        ds = ds.rename(ren)
    return ds

def open_era5_surface_uv(paths: List[Path], time_chunk: int, lat_chunk: int, lon_chunk: int) -> xr.Dataset:
    """Open u10/v10 and u100/v100 for all years; merge into one dataset (dask-chunked)."""
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    bk = {"indexpath": ""}  # avoid on-disk index files for cfgrib
    chunks = {"time": time_chunk}

    log(f"Opening {len(paths)} GRIBs with time chunks={time_chunk} (lazy)...")
    try:
        ds_try = xr.open_mfdataset(
            [str(p) for p in paths],
            engine="cfgrib",
            combine="by_coords",
            chunks=chunks,
            backend_kwargs=bk,
            decode_timedelta=True,
        )
        ds_try = _rename_uv_vars(ds_try)
        have = {k for k in ["u10", "v10", "u100", "v100"] if k in ds_try.data_vars}
        if have == {"u10", "v10", "u100", "v100"}:
            ds = ds_try
        else:
            raise KeyError("Not all u10/v10/u100/v100 present; trying filtered opens.")
    except Exception:
        # Filtered opens per variable, then merge
        def open_one(short):
            return xr.open_mfdataset(
                [str(p) for p in paths],
                engine="cfgrib",
                combine="by_coords",
                chunks=chunks,
                backend_kwargs={**bk, "filter_by_keys": {"shortName": short}},
                decode_timedelta=True,
            )

        d10u = open_one("10u"); d10v = open_one("10v")
        d100u = open_one("100u"); d100v = open_one("100v")

        d10u = _rename_uv_vars(d10u); d10v = _rename_uv_vars(d10v)
        d100u = _rename_uv_vars(d100u); d100v = _rename_uv_vars(d100v)

        ds = xr.merge([d10u, d10v, d100u, d100v], compat="override")

    # Normalize coordinate names
    if "latitude" not in ds.coords and "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "longitude" not in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lon": "longitude"})

    # Add spatial chunking too
    ds = ds.chunk({"time": time_chunk, "latitude": lat_chunk, "longitude": lon_chunk})

    log(f"Dataset sizes: {dict(ds.sizes)}; chunks(time={time_chunk}, lat={lat_chunk}, lon={lon_chunk})")
    for k in ["u10", "v10", "u100", "v100"]:
        if k not in ds:
            raise KeyError(f"Missing variable {k} in merged dataset.")
    return ds

# ---- CORE CALCULATIONS ----
def compute_alpha(spd10: xr.DataArray, spd100: xr.DataArray) -> xr.DataArray:
    """Return alpha = ln(ws10/ws100) / ln(10/100), with fallback where invalid (alpha=1/7)."""
    denom = math.log(10.0 / 100.0)  # negative
    valid = (spd10 > 0) & (spd100 > 0)
    # Mask invalid ratios before log
    ratio = xr.where(valid, spd10 / spd100, np.nan)
    alpha_raw = np.log(ratio) / denom  # NumPy ufunc -> Dask-aware
    alpha = xr.where(np.isfinite(alpha_raw), alpha_raw, ALPHA_FALLBACK)
    return alpha.astype("float32")


def speed_at_height_from_alpha(spd100: xr.DataArray, alpha: xr.DataArray, z_m: float) -> xr.DataArray:
    if abs(z_m - 100.0) < 1e-6:
        return spd100.astype("float32")
    factor = ((z_m / 100.0) ** alpha).astype("float32")
    # Using magnitude scaling (speed100 * factor) is equivalent to hypot(u100*factor, v100*factor)
    return (spd100 * factor).astype("float32")

def quantiles_over_time(da: xr.DataArray) -> xr.DataArray:
    """Compute q0..q100 over time -> DataArray(quantile, latitude, longitude)."""
    qs = np.linspace(0.0, 1.0, 101, dtype="float64")
    q = da.quantile(q=qs, dim="time", skipna=True, method="linear")
    # Replace float probabilities with integer 0..100 labels
    q = q.assign_coords(quantile=(np.arange(101, dtype="int16")))
    return q.astype("float32")

def write_quantiles_csv(q_da: xr.DataArray, out_csv: Path) -> None:
    """
    q_da: DataArray(quantile, latitude, longitude) with quantile coords 0..100 (ints)
    Writes CSV with columns lat, lon, q000..q100, lon wrapped to [-180, 180).
    """
    # Stack lat/lon and pivot quantiles to columns
    q_flat = q_da.stack(point=("latitude", "longitude")).transpose("point", "quantile")
    df = q_flat.to_pandas()  # index=MultiIndex(lat,lon), columns=0..100
    # Build output columns
    new_cols = [f"q{int(c):03d}" for c in df.columns.to_list()]
    df.columns = new_cols
    df = df.reset_index().rename(columns={"latitude": "lat", "longitude": "lon"})
    # Wrap longitudes to [-180, 180)
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    # Lightweight types
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")
    # Write
    df.to_csv(out_csv, index=False)

# ---- PIPELINE ----
def main() -> None:
    ap = argparse.ArgumentParser(
        description="ERA5 grid-wide wind-speed quantiles at multiple heights."
    )
    ap.add_argument("--era5-dir", type=Path, required=True,
                    help="Directory containing ERA5 GRIB files.")
    ap.add_argument("--filename-pattern", type=str, default=DEFAULT_FILENAME_PATTERN,
                    help="GRIB filename pattern with {year} placeholder.")
    ap.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR,
                    help="First year to process (inclusive).")
    ap.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR,
                    help="Last year to process (inclusive).")
    ap.add_argument("--heights", type=str, default=DEFAULT_OUT_HEIGHTS,
                    help="Comma-separated output heights in meters.")
    ap.add_argument("--out-pattern", type=str, default=DEFAULT_OUT_CSV_PATTERN,
                    help="Output CSV filename pattern with {height_m} placeholder.")
    ap.add_argument("--time-chunk", type=int,
                    default=int(os.getenv("ERA5_TIME_CHUNK", "8928")),
                    help="Dask time chunk size.")
    ap.add_argument("--lat-chunk", type=int,
                    default=int(os.getenv("ERA5_LAT_CHUNK", "64")),
                    help="Dask latitude chunk size.")
    ap.add_argument("--lon-chunk", type=int,
                    default=int(os.getenv("ERA5_LON_CHUNK", "64")),
                    help="Dask longitude chunk size.")
    args = ap.parse_args()

    out_heights = [float(h.strip()) for h in args.heights.split(",")]
    years = list(range(args.start_year, args.end_year + 1))
    paths = grib_paths(args.era5_dir, args.filename_pattern, years)
    log(f"Verifying {len(paths)} ERA5 files...")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        for m in missing:
            log(f"[MISSING] {m}")
        raise FileNotFoundError("One or more ERA5 GRIBs missing.")

    ds = open_era5_surface_uv(paths, args.time_chunk, args.lat_chunk, args.lon_chunk)

    # Speeds at native levels
    spd10  = np.hypot(ds["u10"],  ds["v10"]).astype("float32")
    spd100 = np.hypot(ds["u100"], ds["v100"]).astype("float32")

    # alpha field (time x lat x lon)
    log("Computing alpha (power-law exponent between 10 m and 100 m)...")
    alpha = compute_alpha(spd10, spd100)

    # Quantiles per requested height
    for z in out_heights:
        t0 = time.perf_counter()
        log(f"[{z:.0f} m] Computing speed field...")
        spd_z = speed_at_height_from_alpha(spd100, alpha, z)

        log(f"[{z:.0f} m] Computing q0..q100 over time (lazy)...")
        q_da = quantiles_over_time(spd_z)  # (quantile, latitude, longitude)

        out_csv = Path(args.out_pattern.format(height_m=z))
        log(f"[{z:.0f} m] Writing {out_csv} (this triggers compute)...")
        write_quantiles_csv(q_da, out_csv)

        dt = time.perf_counter() - t0
        log(f"[{z:.0f} m] Done in {dt:0.1f}s -> {out_csv}")

    try:
        ds.close()
    except Exception:
        pass

    log("[DONE] All height CSVs written.")

if __name__ == "__main__":
    xr.set_options(keep_attrs=True)
    main()
