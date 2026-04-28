#!/usr/bin/env python3
"""
Vectorized ERA5 quantiles at per-site height (2007-2024), using local GRIBs.
Vertical interpolation uses a power law between 10 m and 100 m.

Inputs
------
gold_standard_quantiles.csv (must contain: station_id, name, lat, lon, elev_m, height_m)
<era5-dir>/conus-YYYY-hourly.grib  (pass via --era5-dir)

Output
------
era5_quantiles_gold_standard_2007_2024.csv   # one row per (station_id, height_m)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from wem.utils.logging import log
from wem.utils.sites import load_gs_sites, already_done_gs

# ── CONFIG DEFAULTS ──────────────────────────────────────────
ALPHA_FALLBACK = 1.0 / 7.0  # neutral power-law exponent

# ── ERA5 OPEN/MERGE ──────────────────────────────────────────
def grib_paths(era5_dir: Path, pattern: str, years: List[int]) -> List[Path]:
    return [era5_dir / pattern.format(year=y) for y in years]

def _rename_uv_vars(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "10u" in ds:  ren["10u"] = "u10"
    if "10v" in ds:  ren["10v"] = "v10"
    if "100u" in ds: ren["100u"] = "u100"
    if "100v" in ds: ren["100v"] = "v100"
    if ren:
        ds = ds.rename(ren)
    return ds

def open_era5_surface_uv(paths: List[Path], time_chunk: int) -> xr.Dataset:
    """Open u10/v10 and u100/v100 for all years; merge into one dataset (dask-chunked)."""
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    bk = {"indexpath": ""}  # avoid on-disk index files
    chunks = {"time": time_chunk}

    log(f"Opening {len(paths)} GRIBs with chunks time={time_chunk} (lazy)...")
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

    log(f"Dataset sizes: {dict(ds.sizes)}")
    # Basic sanity
    for k in ["u10", "v10", "u100", "v100"]:
        if k not in ds:
            raise KeyError(f"Missing variable {k} in merged dataset.")
    return ds

def wrap_lons_if_needed(lons_deg: np.ndarray, ds: xr.Dataset) -> np.ndarray:
    """ERA5 longitudes are often 0..360. If so, wrap input lons to 0..360."""
    try:
        lon_coord = ds["longitude"]
        lon_max = float(lon_coord.max())
        if lon_max > 180.0:
            l = lons_deg % 360.0
            l[l < 0] += 360.0
            return l
    except Exception:
        pass
    return lons_deg

# ── CORE: VECTORIZED BATCH PROCESSING ────────────────────────
def batch_iter(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)

def quantiles_and_counts_for_batch(
    ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray, heights_m: np.ndarray,
    interp_method: str,
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """
    Vectorized pipeline:
      1) Bilinear interp u10,v10,u100,v100 to (site) -> (time, site)
      2) Compute speed-based alpha between 10 m and 100 m (fallback 1/7 where invalid)
      3) Apply power law to u/v (using common alpha) to target heights per site
      4) Quantiles (q000..q100) over time per site; counts of finite samples
    """
    # Site arrays
    lat_da = xr.DataArray(lats, dims="site")
    lon_da = xr.DataArray(lons, dims="site")

    # 1) Bilinear interpolation at the sites
    u10 = ds["u10"].interp(latitude=lat_da, longitude=lon_da, method=interp_method)
    v10 = ds["v10"].interp(latitude=lat_da, longitude=lon_da, method=interp_method)
    u100 = ds["u100"].interp(latitude=lat_da, longitude=lon_da, method=interp_method)
    v100 = ds["v100"].interp(latitude=lat_da, longitude=lon_da, method=interp_method)

    # 2) speed-based alpha (time, site)
    spd10 = xr.apply_ufunc(np.hypot, u10, v10, dask="parallelized")
    spd100 = xr.apply_ufunc(np.hypot, u100, v100, dask="parallelized")

    # Load to NumPy for the vertical step (keeps memory predictable)
    t0 = time.perf_counter()
    u10n, v10n, u100n, v100n = (u10.load().values, v10.load().values,
                                u100.load().values, v100.load().values)
    spd10n, spd100n = (spd10.load().values, spd100.load().values)
    t1 = time.perf_counter()

    # shapes: (T, S)
    T, S = u10n.shape
    z = heights_m.reshape(1, S).astype("float64")  # (1,S)

    # alpha = ln(ws10/ws100)/ln(10/100), fallback where invalid
    denom = math.log(10.0/100.0)  # negative
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_raw = np.log(spd10n / spd100n) / denom
    invalid = ~np.isfinite(alpha_raw) | (spd10n <= 0.0) | (spd100n <= 0.0)
    alpha = np.where(invalid, ALPHA_FALLBACK, alpha_raw).astype("float64")

    # 3) apply power law to u,v using common alpha from speeds (more stable)
    factor = (z / 100.0) ** alpha  # (T,S)
    # use u100/v100 as reference (could also use 10 m; using upper reduces extrapolation for z>10)
    u_z = (u100n * factor).astype("float32")
    v_z = (v100n * factor).astype("float32")
    spd_z = np.hypot(u_z, v_z, dtype="float32")  # (T,S)

    # 4) counts and quantiles
    counts = np.isfinite(spd_z).sum(axis=0).astype(int)
    q_perc = np.arange(101)  # 0..100
    qvals = np.nanpercentile(spd_z, q=q_perc, axis=0, method="linear")  # (101, S)

    qcols = [f"q{p:03d}" for p in q_perc]
    q_df = pd.DataFrame(qvals.T, columns=qcols)  # site x quantile

    return q_df, counts, (t1 - t0)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vectorized ERA5 quantiles at per-site height (GS cohort)."
    )
    ap.add_argument("--sites", type=Path, default=Path("gold_standard_quantiles_ozark.csv"),
                    help="Sites CSV with station_id, name, lat, lon, elev_m, height_m.")
    ap.add_argument("--era5-dir", type=Path, required=True,
                    help="Directory with ERA5 GRIB files.")
    ap.add_argument("--filename-pattern", type=str, default="conus-{year}-hourly.grib",
                    help="GRIB filename pattern with {year} placeholder.")
    ap.add_argument("--out", type=Path, default=Path("era5_quantiles_gold_standard_2007_2024_ozark.csv"),
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

    sites_all = load_gs_sites(LOCKED_SITES_CSV)
    done = already_done_gs(OUT_CSV)

    # Keep rows not yet written (per station_id+height_m)
    mask_done = [(str(s), float(h)) in done for s, h in zip(sites_all["station_id"], sites_all["height_m"])]
    sites = sites_all.loc[~pd.Series(mask_done).values].reset_index(drop=True)
    total_sites = len(sites_all)
    todo_sites = len(sites)
    log(f"[INFO] Loaded {total_sites} (station_id,height) pairs; {len(done)} already done; {todo_sites} to process.")
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

    ds = open_era5_surface_uv(paths, TIME_CHUNK)

    # Stable output header
    meta_cols = [
        "station_id", "name", "lat", "lon", "elev_m",
        "dataset", "height_m", "interp", "years", "n_hours", "processed_utc",
    ]
    qcols = [f"q{q:03d}" for q in range(101)]
    header_cols = meta_cols + qcols

    appended = 0
    nbatches = (todo_sites + BATCH_SIZE - 1) // BATCH_SIZE
    lon_coord_max = float(ds["longitude"].max())
    log(f"[INFO] Processing {todo_sites} sites in {nbatches} batches of {BATCH_SIZE} (interp={INTERP_METHOD}); "
        f"ERA5 lon max={lon_coord_max:.1f}")

    for bidx, (i0, i1) in enumerate(batch_iter(todo_sites, BATCH_SIZE), start=1):
        batch = sites.iloc[i0:i1].copy()
        lats = batch["lat"].to_numpy(float)
        lons = batch["lon"].to_numpy(float)
        lons = wrap_lons_if_needed(lons, ds)  # handle 0..360 grids
        heights = batch["height_m"].to_numpy(float)

        log(f"[BATCH {bidx}/{nbatches}] sites {i0}-{i1-1} (n={len(batch)}) -> interp u10/v10/u100/v100 & load...")
        q_df, counts, load_s = quantiles_and_counts_for_batch(ds, lats, lons, heights, INTERP_METHOD)
        zero_mask = counts == 0
        n_zero = int(zero_mask.sum())
        log(f"[BATCH {bidx}/{nbatches}] loaded in {load_s:0.2f}s; {n_zero} site(s) had 0 valid hours.")

        ts = pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"
        rows: List[Dict[str, object]] = []
        for i, (_, r) in enumerate(batch.iterrows()):
            if zero_mask[i]:
                log(f"[SKIP] {r['station_id']} (h={r['height_m']}) outside domain (0 valid hours).")
                continue

            row: Dict[str, object] = {
                "station_id": str(r["station_id"]),
                "name":       str(r.get("name", r["station_id"])),
                "lat":        float(r["lat"]),
                "lon":        float(r["lon"]),
                "elev_m":     (float(r["elev_m"]) if pd.notna(r["elev_m"]) else None),
                "dataset":    "ERA5",
                "height_m":   float(r["height_m"]),
                "interp":     f"{INTERP_METHOD}+powerlaw(10-100)",
                "years":      f"{START_YEAR}-{END_YEAR}",
                "n_hours":    int(counts[i]),
                "processed_utc": ts,
            }
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
    xr.set_options(keep_attrs=True)
    main()
