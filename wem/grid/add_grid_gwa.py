#!/usr/bin/env python3
"""
Add Global Wind Atlas (GWA) mean-speed to an inference table, efficiently.

- Samples each GWA raster once per UNIQUE (lon,lat)
- Computes gwa_interp once per UNIQUE (lon,lat,height_m) using power-law fit
- Merges results back to all rows with the same keys

Input must have columns:
  lat, lon, height_m  (others are passed through unchanged)

Outputs columns added:
  gwa_10, gwa_50, gwa_100, gwa_150 (if available), and gwa_interp

Usage
-----
python add_grid_gwa.py \\
  --in inference_table.csv \\
  --out inference_table_with_gwa.csv \\
  --gwa10 /path/GWA_US_10m.tif \\
  --gwa50 /path/GWA_US_50m.tif \\
  --gwa100 /path/GWA_US_100m.tif \\
  --gwa150 /path/GWA_US_150m.tif
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform

from wem.utils.logging import log

# ------------------ raster sampling ------------------
def _sample_one(ds: rasterio.io.DatasetReader,
                lons: np.ndarray,
                lats: np.ndarray,
                chunksize: int = 200_000) -> np.ndarray:
    """Sample a single-band raster at lon/lat coords; returns float32 with NaN for nodata."""
    N = lons.size
    out = np.full(N, np.nan, dtype="float32")

    # Transform to dataset CRS if needed
    if ds.crs is None or ds.crs.to_epsg() == 4326:
        xs, ys = lons, lats
    else:
        xs, ys = rio_transform("EPSG:4326", ds.crs, lons.tolist(), lats.tolist())

    nodata = ds.nodata
    for i in range(0, N, chunksize):
        j = min(i + chunksize, N)
        coords = list(zip(xs[i:j], ys[i:j]))
        vals = np.array([v[0] for v in ds.sample(coords)], dtype="float32")
        bad = ~np.isfinite(vals)
        if nodata is not None:
            bad |= (vals == nodata)
        vals[bad] = np.nan
        out[i:j] = vals
    return out

def sample_gwa_heights(gwa_paths: Dict[int, Path],
                       uniq_lon: np.ndarray,
                       uniq_lat: np.ndarray) -> Dict[int, np.ndarray]:
    """Sample provided GWA rasters (subset of {10,50,100,150}) at UNIQUE lon/lat once."""
    out: Dict[int, np.ndarray] = {}
    for z, p in sorted(gwa_paths.items()):
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            log(f"[WARN] GWA file missing for {z} m: {p}")
            continue
        log(f"[GWA] Sampling {z} m raster over {len(uniq_lon):,} unique coords: {p.name}")
        with rasterio.open(p) as ds:
            out[z] = _sample_one(ds, uniq_lon, uniq_lat)
    if not out:
        raise SystemExit("No valid GWA rasters were provided/found.")
    return out

# ------------- vertical interpolation (power law) -------------
_ZS = np.array([10.0, 50.0, 100.0, 150.0], dtype="float64")
_LNZ = np.log(_ZS)[None, :]  # shape (1,4)

def powerlaw_interpolate_rows(
    target_h: np.ndarray,
    u10: Optional[np.ndarray],
    u50: Optional[np.ndarray],
    u100: Optional[np.ndarray],
    u150: Optional[np.ndarray],
) -> np.ndarray:
    """Vectorized per-row power-law fit on any subset of {10,50,100,150}; returns float32."""
    N = target_h.size
    cols: List[np.ndarray] = []
    for arr in (u10, u50, u100, u150):
        if arr is None:
            cols.append(np.full(N, np.nan, dtype="float64"))
        else:
            cols.append(arr.astype("float64", copy=False))
    U = np.stack(cols, axis=1)  # (N,4)

    M = np.isfinite(U) & (U > 0.0)
    k = M.sum(axis=1)

    Y = np.full_like(U, 0.0, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        Y[M] = np.log(U[M])

    sumx  = (_LNZ * M).sum(axis=1)
    sumy  = (Y * M).sum(axis=1)
    sumxy = (_LNZ * Y * M).sum(axis=1)
    sumx2 = ((_LNZ ** 2) * M).sum(axis=1)

    denom = k * sumx2 - (sumx ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (k * sumxy - sumx * sumy) / denom
        mean_x = np.divide(sumx, k, out=np.zeros_like(sumx), where=(k > 0))
        mean_y = np.divide(sumy, k, out=np.zeros_like(sumy), where=(k > 0))
        a = mean_y - alpha * mean_x
        A = np.exp(a)

    zt = np.asarray(target_h, dtype="float64")
    zt = np.where(np.isfinite(zt) & (zt > 0), zt, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        u_fit = A * np.power(zt, alpha)

    need_fallback = (k < 2)
    if np.any(need_fallback):
        D = np.abs(_ZS[None, :] - zt[:, None])
        D[~M] = np.inf
        jmin = D.argmin(axis=1)
        fallback = np.where(
            np.isfinite(U[np.arange(N), jmin]),
            U[np.arange(N), jmin],
            np.nan
        )
        u_fit = np.where(need_fallback, fallback, u_fit)

    return u_fit.astype("float32")

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Add GWA (gwa_interp) to an inference table with unique-key caching.")
    ap.add_argument("--in",  dest="infile",  type=Path, required=True)
    ap.add_argument("--out", dest="outfile", type=Path, required=True)

    ap.add_argument("--gwa10",  type=Path, required=True, help="GWA 10 m GeoTIFF")
    ap.add_argument("--gwa50",  type=Path, required=True, help="GWA 50 m GeoTIFF")
    ap.add_argument("--gwa100", type=Path, required=True, help="GWA 100 m GeoTIFF")
    ap.add_argument("--gwa150", type=Path, required=True, help="GWA 150 m GeoTIFF")
    args = ap.parse_args()

    # ---- Load table ----
    log(f"[INFO] Loading inference table: {args.infile}")
    if args.infile.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(args.infile)
    else:
        df = pd.read_csv(args.infile, low_memory=False)

    # Required cols
    for c in ("lat", "lon", "height_m"):
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # Clean
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)

    # ---- Unique coordinate sampling (lon,lat) ----
    uniq_xy = df[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    log(f"[INFO] Unique (lon,lat) to sample: {len(uniq_xy):,}")

    gwa_paths = {10: args.gwa10, 50: args.gwa50, 100: args.gwa100, 150: args.gwa150}
    samples_xy = sample_gwa_heights(
        gwa_paths,
        uniq_xy["lon"].to_numpy("float64"),
        uniq_xy["lat"].to_numpy("float64"),
    )

    # Attach sampled heights to uniq_xy (only those available)
    for z, arr in samples_xy.items():
        uniq_xy[f"gwa_{z}"] = arr

    # ---- Unique triples (lon,lat,height_m) for interpolation ----
    uniq_xyz = df[["lon", "lat", "height_m"]].drop_duplicates().reset_index(drop=True)
    log(f"[INFO] Unique (lon,lat,height_m) triples: {len(uniq_xyz):,}")

    # Merge sampled values from uniq_xy to uniq_xyz
    keep_cols = ["lon", "lat"] + [c for c in ["gwa_10", "gwa_50", "gwa_100", "gwa_150"] if c in uniq_xy.columns]
    uniq_xyz = uniq_xyz.merge(uniq_xy[keep_cols], on=["lon", "lat"], how="left")

    # Compute gwa_interp for each unique triple
    u10  = uniq_xyz.get("gwa_10",  pd.Series(np.nan, index=uniq_xyz.index, dtype="float32")).to_numpy("float32")
    u50  = uniq_xyz.get("gwa_50",  pd.Series(np.nan, index=uniq_xyz.index, dtype="float32")).to_numpy("float32")
    u100 = uniq_xyz.get("gwa_100", pd.Series(np.nan, index=uniq_xyz.index, dtype="float32")).to_numpy("float32")
    u150 = uniq_xyz.get("gwa_150", pd.Series(np.nan, index=uniq_xyz.index, dtype="float32")).to_numpy("float32")

    log("[INFO] Computing vertical interpolation (power law) on unique triples ...")
    uniq_xyz["gwa_interp"] = powerlaw_interpolate_rows(
        uniq_xyz["height_m"].to_numpy("float64"),
        u10, u50, u100, u150
    )

    # ---- Merge back to full table (broadcast to duplicates) ----
    out = df.merge(
        uniq_xyz[["lon", "lat", "height_m"] + [c for c in ["gwa_10","gwa_50","gwa_100","gwa_150"] if c in uniq_xyz.columns] + ["gwa_interp"]],
        on=["lon", "lat", "height_m"],
        how="left"
    )

    # ---- Save ----
    log(f"[INFO] Writing -> {args.outfile}")
    if args.outfile.suffix.lower() in (".parquet", ".pq"):
        out.to_parquet(args.outfile, index=False)
    else:
        out.to_csv(args.outfile, index=False)

    nn = int(np.isfinite(out["gwa_interp"]).sum())
    log(f"[INFO] Done. Rows: {len(out):,} | non-NaN gwa_interp: {nn:,} | "
        f"unique coords: {len(uniq_xy):,} | unique triples: {len(uniq_xyz):,}")

if __name__ == "__main__":
    main()
