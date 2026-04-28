#!/usr/bin/env python3
"""
Vectorized ERA5 quantiles at 10 m for many sites (2007-2024), using local GRIBs.

Inputs
------
all_sites_quantiles_2007_2024.csv
<era5-dir>/conus-YYYY-hourly.grib  (pass via --era5-dir)

Output
------
era5_quantiles_2007_2024.csv   # one row per site: metadata + q000..q100
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from wem.utils.logging import log
from wem.utils.sites import load_sites, already_done

# ── ERA5 OPEN/MERGE ──────────────────────────────────────────
def grib_paths(era5_dir: Path, pattern: str, years: List[int]) -> List[Path]:
    return [era5_dir / pattern.format(year=y) for y in years]

def open_era5_10uv(paths: List[Path], time_chunk: int) -> xr.Dataset:
    """Open all years; ensure u10 & v10 present; return merged dataset (dask-chunked)."""
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    bk = {"indexpath": ""}  # avoid on-disk index files
    chunks = {"time": time_chunk}

    log(f"Opening {len(paths)} GRIBs with chunks time={time_chunk} (lazy)...")
    t0 = time.perf_counter()
    # Try one-shot open (both vars together)
    try:
        ds_try = xr.open_mfdataset(
            [str(p) for p in paths],
            engine="cfgrib",
            combine="by_coords",
            chunks=chunks,
            backend_kwargs=bk,
            decode_timedelta=True,   # silence FutureWarning from xarray
        )
        have_u = any(k in ds_try.data_vars for k in ("u10", "10u"))
        have_v = any(k in ds_try.data_vars for k in ("v10", "10v"))
        if have_u and have_v:
            ds = ds_try
        else:
            raise KeyError("u10/v10 not both present; trying filtered opens.")
    except Exception:
        # Open filtered views and merge
        ds_u = xr.open_mfdataset(
            [str(p) for p in paths],
            engine="cfgrib",
            combine="by_coords",
            chunks=chunks,
            backend_kwargs={**bk, "filter_by_keys": {"shortName": "10u"}},
            decode_timedelta=True,
        )
        ds_v = xr.open_mfdataset(
            [str(p) for p in paths],
            engine="cfgrib",
            combine="by_coords",
            chunks=chunks,
            backend_kwargs={**bk, "filter_by_keys": {"shortName": "10v"}},
            decode_timedelta=True,
        )
        if "10u" in ds_u:
            ds_u = ds_u.rename({"10u": "u10"})
        if "10v" in ds_v:
            ds_v = ds_v.rename({"10v": "v10"})
        ds = xr.merge([ds_u, ds_v], compat="override")

    # Normalize coordinate names for interp()
    if "latitude" not in ds.coords and "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "longitude" not in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lon": "longitude"})

    t1 = time.perf_counter()
    # Use sizes mapping (avoids dims FutureWarning)
    log(f"Opened lazily in {t1 - t0:0.2f}s. Dataset sizes: {dict(ds.sizes)}")
    return ds

# ── CORE: VECTORIZED BATCH PROCESSING ────────────────────────
def batch_iter(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)

def quantiles_and_counts_for_batch(
    ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray, interp_method: str,
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """
    Interpolate u10,v10 for a batch of sites (vectorized), compute:
      - q000..q100 quantiles over time (ignoring NaNs),
      - non-NaN hour counts per site.
    Returns (q_df, counts, load_seconds) where q_df shape is (site, 101).
    """
    site = xr.DataArray(np.arange(lats.size), dims="site")
    lat_da = xr.DataArray(lats, dims="site")
    lon_da = xr.DataArray(lons, dims="site")

    # Vectorized interpolation (time x site)
    u_pt = ds["u10"].interp(latitude=lat_da, longitude=lon_da, method=interp_method)
    v_pt = ds["v10"].interp(latitude=lat_da, longitude=lon_da, method=interp_method)

    # Speed (keep float32 to manage memory)
    spd = xr.apply_ufunc(np.hypot, u_pt, v_pt, dask="parallelized").astype("float32")

    # Load this batch into memory, compute quantiles and counts
    t0 = time.perf_counter()
    spd_np = spd.load().values  # shape (T, S)
    t1 = time.perf_counter()

    # Counts of valid hours per site
    counts = np.isfinite(spd_np).sum(axis=0).astype(int)

    # 101 integer-percent quantiles per site (avoids float rounding label dupes)
    # Note: np.nanpercentile returns NaN for all-NaN slices (we'll handle later)
    q_perc = np.arange(101)  # 0..100
    qvals = np.nanpercentile(spd_np, q=q_perc, axis=0, method="linear")  # (101, S)

    qcols = [f"q{p:03d}" for p in q_perc]
    q_df = pd.DataFrame(qvals.T, columns=qcols)  # site x quantile

    return q_df, counts, (t1 - t0)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vectorized ERA5 quantiles at 10 m for ASOS sites."
    )
    ap.add_argument("--sites", type=Path, default=Path("all_sites_quantiles_2007_2024.csv"),
                    help="Sites CSV with station_id, name, lat, lon, elev_m.")
    ap.add_argument("--era5-dir", type=Path, required=True,
                    help="Directory with ERA5 GRIB files.")
    ap.add_argument("--filename-pattern", type=str, default="conus-{year}-hourly.grib",
                    help="GRIB filename pattern with {year} placeholder.")
    ap.add_argument("--out", type=Path, default=Path("era5_quantiles_2007_2024.csv"),
                    help="Output CSV path.")
    ap.add_argument("--start-year", type=int, default=2007, help="First year to process.")
    ap.add_argument("--end-year", type=int, default=2024, help="Last year to process.")
    ap.add_argument("--interp-method", type=str, default="linear",
                    help="Interpolation method for xarray.interp() (linear or nearest).")
    ap.add_argument("--batch-size", type=int,
                    default=int(os.getenv("ERA5_BATCH_SIZE", "200")),
                    help="Sites per vectorized interpolation batch.")
    ap.add_argument("--time-chunk", type=int,
                    default=int(os.getenv("ERA5_TIME_CHUNK", "8928")),
                    help="Dask time chunk size.")
    args = ap.parse_args()

    LOCKED_SITES_CSV = args.sites
    ERA5_DIR = args.era5_dir
    FILENAME_PATTERN = args.filename_pattern
    OUT_CSV = args.out
    START_YEAR = args.start_year
    END_YEAR = args.end_year
    INTERP_METHOD = args.interp_method
    BATCH_SIZE = args.batch_size
    TIME_CHUNK = args.time_chunk

    sites_all = load_sites(LOCKED_SITES_CSV)
    done_ids = already_done(OUT_CSV)

    sites = sites_all[~sites_all["station_id"].isin(done_ids)].reset_index(drop=True)
    total_sites = len(sites_all)
    todo_sites = len(sites)
    log(f"[INFO] Loaded {total_sites} sites; {len(done_ids)} already done; {todo_sites} to process.")
    if sites.empty:
        log("[INFO] Nothing to do; exiting.")
        return

    years = list(range(START_YEAR, END_YEAR + 1))
    paths = grib_paths(ERA5_DIR, FILENAME_PATTERN, years)
    log(f"[INFO] Verifying {len(paths)} ERA5 files...")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        for m in missing:
            log(f"[MISSING] {m}")
        raise FileNotFoundError("One or more ERA5 GRIBs missing.")

    ds = open_era5_10uv(paths, TIME_CHUNK)

    # Stable output header
    meta_cols = [
        "station_id", "name", "lat", "lon", "elev_m",
        "dataset", "height_m", "interp", "years", "n_hours", "processed_utc",
    ]
    qcols = [f"q{q:03d}" for q in range(101)]
    header_cols = meta_cols + qcols

    appended = 0
    nbatches = (todo_sites + BATCH_SIZE - 1) // BATCH_SIZE
    log(f"[INFO] Processing {todo_sites} sites in {nbatches} batches of {BATCH_SIZE} (interp={INTERP_METHOD})")

    for bidx, (i0, i1) in enumerate(batch_iter(todo_sites, BATCH_SIZE), start=1):
        batch = sites.iloc[i0:i1].copy()
        lats = batch["lat"].to_numpy(float)
        lons = batch["lon"].to_numpy(float)

        log(f"[BATCH {bidx}/{nbatches}] sites {i0}-{i1-1} (n={len(batch)}) -> interpolate & load...")
        q_df, counts, load_s = quantiles_and_counts_for_batch(ds, lats, lons, INTERP_METHOD)
        zero_mask = counts == 0
        n_zero = int(zero_mask.sum())
        log(f"[BATCH {bidx}/{nbatches}] loaded in {load_s:0.2f}s; {n_zero} site(s) had 0 valid hours.")

        ts = pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"
        rows: List[Dict[str, object]] = []
        for i, (_, r) in enumerate(batch.iterrows()):
            # Skip sites totally outside the ERA5 CONUS tile
            if zero_mask[i]:
                log(f"[SKIP] {r['station_id']} outside domain (0 valid hours).")
                continue

            row: Dict[str, object] = {
                "station_id": str(r["station_id"]),
                "name":       str(r.get("name", r["station_id"])),
                "lat":        float(r["lat"]),
                "lon":        float(r["lon"]),
                "elev_m":     (float(r["elev_m"]) if pd.notna(r["elev_m"]) else None),
                "dataset":    "ERA5",
                "height_m":   10,
                "interp":     INTERP_METHOD,
                "years":      f"{START_YEAR}-{END_YEAR}",
                "n_hours":    int(counts[i]),
                "processed_utc": ts,
            }
            # attach quantiles via .at (guaranteed scalar)
            for c in qcols:
                val = q_df.at[i, c]
                row[c] = float(val) if pd.notna(val) else None
            rows.append(row)

        if not rows:
            log(f"[BATCH {bidx}/{nbatches}] no rows to append (all sites skipped).")
            continue

        out_df = pd.DataFrame(rows, columns=header_cols)
        header_needed = not OUT_CSV.exists()
        out_df.to_csv(OUT_CSV, mode="a", header=header_needed, index=False)
        appended += len(rows)
        log(f"[BATCH {bidx}/{nbatches}] appended {len(rows)} rows (total appended this run: {appended})")

    # Close dataset handles explicitly
    try:
        ds.close()
    except Exception:
        pass

    log(f"[DONE] Wrote {OUT_CSV} ({appended} new rows).")

if __name__ == "__main__":
    # Silence timedelta warning specifically through xarray open kwargs above.
    # Keep attrs (harmless); users can tweak defaults if needed.
    xr.set_options(keep_attrs=True)
    main()
