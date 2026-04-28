#!/usr/bin/env python3
"""
ERA5-grid -> WTK wind-speed quantiles at ERA5 heights, tile-optimized with I/O optimizations.

- Spatial:  IDW (K=4) on u,v from WTK grid -> ERA5 grid points.
- Vertical: Power-law between bracketing WTK heights (fallback alpha = 1/7).
- Years:    2007-2013 (local WTK CONUS files: wtk_conus_YYYY.h5 or conus_YYYY.h5).
- Tiling:   2-D Lambert Conformal tiling to bound memory & I/O.
- Output:   One CSV/Parquet per tile; each row = (grid_id, height_m, q000..q100).

Examples
--------
# List tile IDs and exit
python grid_wtk.py --era5-grid era5_grid.csv --list-tiles > tiles.txt

# Run a single tile
python grid_wtk.py --era5-grid era5_grid.csv \\
  --wtk-dir /datasets/WIND/WTK --tile-only 3 --out-dir out_wtk --format parquet
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
from tqdm.auto import tqdm

from wem.utils.logging import log
from wem.utils.spatial import to_xy_lcc
from wem.grid._grid_shared import Tiles, NeighborPack, load_era5_grid, make_tiles, contiguous_runs
from wem.utils.wind import uv_from_ws_wd
from wem.utils.quantiles import quantile_block
from wem.utils.power_law import bracket_for_height
from wem.constants import WTK_HEIGHTS

try:
    import psutil  # optional, for RSS reporting
except Exception:
    psutil = None

# -------------------- Config defaults --------------------
YEARS       = list(range(2007, 2014))  # inclusive 2007-2013
K_NEIGH     = 4
IDW_POWER   = 1.0
FALLBACK_ALPHA = 1.0/7.0
LOG_PREFIX  = "[WTK]"

def rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return None

# -------------------- Helpers --------------------
def get_scale(ds: h5py.Dataset) -> tuple[float, float]:
    """Read scale_factor/add_offset if present; else (1.0, 0.0)."""
    def _get(name: str) -> Optional[float]:
        if name in ds.attrs:
            val = ds.attrs[name]
        elif name.encode() in ds.attrs:
            val = ds.attrs[name.encode()]
        else:
            return None
        try:
            return float(val)
        except Exception:
            try:
                return float(np.array(val).astype("float64"))
            except Exception:
                return None
    sf = _get("scale_factor")
    ao = _get("add_offset")
    return (sf if sf is not None else 1.0, ao if ao is not None else 0.0)

def open_year_file(data_dir: Path, year: int) -> h5py.File:
    for name in (f"wtk_conus_{year}.h5", f"conus_{year}.h5", f"conus-{year}.h5"):
        fp = data_dir / name
        if fp.exists():
            return h5py.File(fp, "r")
    raise FileNotFoundError(f"No local WTK file found for year {year} in {data_dir}")

# -------------------- Neighbors: build weights per tile --------------------
def build_wtk_tree(lead: h5py.File) -> tuple[cKDTree, np.ndarray]:
    coords = np.asarray(lead["coordinates"])  # (N,2) [lat, lon]
    lat = coords[:, 0].astype("float64"); lon = coords[:, 1].astype("float64")
    x, y = to_xy_lcc(lon, lat)
    ok = np.isfinite(x) & np.isfinite(y)
    idx_map = np.nonzero(ok)[0].astype(np.int64)
    tree = cKDTree(np.column_stack([x[ok], y[ok]]))
    return tree, idx_map

def neighbor_weights_idw4(
    site_xy: np.ndarray, tree: cKDTree, idx_map: np.ndarray, power: float = IDW_POWER
) -> NeighborPack:
    dd, idxs = tree.query(site_xy, k=K_NEIGH)  # (S,4)
    idxs = idxs.astype(int); dd = dd.astype(float)
    S = dd.shape[0]

    weights = np.empty_like(dd, dtype="float64")
    for i in range(S):
        d = dd[i]
        if np.any(d == 0.0):
            w = np.zeros_like(d); w[np.argmin(d)] = 1.0
            weights[i] = w
        else:
            inv = 1.0 / np.power(d, power)
            weights[i] = inv / inv.sum()

    nbr_cols = idx_map[idxs]  # (S,4) WTK column indices
    uniq_cols = np.unique(nbr_cols.ravel())  # sorted asc
    pos = {int(c): j for j, c in enumerate(uniq_cols)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for s in range(S):
        for n in range(K_NEIGH):
            c = int(nbr_cols[s, n])
            rows.append(pos[c])
            cols.append(s)
            data.append(float(weights[s, n]))
    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(uniq_cols), S), dtype="float64")
    return NeighborPack(uniq_cols=uniq_cols, W=W)

# -------------------- Optimized reads: contiguous blocks --------------------
def read_var_blocks_h5(ds: h5py.Dataset, uniq_cols_sorted: np.ndarray) -> np.ndarray:
    """
    Read (T, K_unique) from an HDF5 dataset using coalesced contiguous column blocks.
    Returns float32 (decoded later by caller if needed).
    """
    uniq_cols_sorted = np.asarray(uniq_cols_sorted, dtype=int)
    if uniq_cols_sorted.size == 0:
        return np.zeros((0, 0), dtype="float32")
    runs = contiguous_runs(uniq_cols_sorted)
    blocks: List[np.ndarray] = []
    for s, e in runs:
        part = ds[:, s:e]                    # (T, e-s)
        blocks.append(np.asarray(part, dtype="float32"))
    return np.concatenate(blocks, axis=1).astype("float32", copy=False)

# -------------------- Core per-tile (optimized) --------------------
def compute_tile_quantiles_optimized(
    tile_idx: np.ndarray,
    era5_df: pd.DataFrame,
    neighbor_pack: NeighborPack,
    year_files: Dict[int, h5py.File],
    target_heights: np.ndarray,
) -> Tuple[pd.DataFrame, dict, List[int]]:
    """
    Optimized per-tile computation:
      - For each YEAR: read each needed native HEIGHT once (contiguous column blocks),
        convert to uK/vK and PROJECT once (K->S). Cache uS/vS per (year,height).
      - For each target height z: re-use cached uS/vS to do vertical interpolation (if needed)
        and compute quantiles.
    Returns (DataFrame, timing_dict, needed_heights).
    """
    sub = era5_df.iloc[tile_idx].reset_index(drop=True)
    site_ids = sub["grid_id"].astype("string").to_numpy()
    S = len(sub)
    uniq_cols = neighbor_pack.uniq_cols  # sorted asc
    W = neighbor_pack.W                  # (K_unique, S), csr

    # Determine native heights needed across all z (avoid duplicate reads)
    needed_heights: List[int] = []
    brackets: Dict[int, Tuple[bool,int,int]] = {}
    for z in map(float, target_heights):
        is_exact, h_lo, h_hi = bracket_for_height(z, WTK_HEIGHTS)
        brackets[int(z)] = (is_exact, h_lo, h_hi)
        if is_exact:
            needed_heights.append(h_lo)
        else:
            needed_heights.extend([h_lo, h_hi])
    needed_heights = sorted(set(needed_heights))

    # Per-year caches: (year,height) -> uS/vS arrays (T,S)
    uS_cache: Dict[tuple[int,int], np.ndarray] = {}
    vS_cache: Dict[tuple[int,int], np.ndarray] = {}

    timing = {"per_year": {}, "quantiles": 0.0, "compute_total": 0.0}
    t_compute0 = time.perf_counter()

    # 1) For each year, pre-read all needed heights once
    for y in YEARS:
        read_s = uv_s = proj_s = 0.0
        fy = year_files[y]
        for h in needed_heights:
            ws_ds = fy[f"windspeed_{h}m"]
            wd_ds = fy[f"winddirection_{h}m"]
            sf_ws, of_ws = get_scale(ws_ds)
            sf_wd, of_wd = get_scale(wd_ds)

            t0 = time.perf_counter()
            ws_raw = read_var_blocks_h5(ws_ds, uniq_cols)  # (T,K)
            wd_raw = read_var_blocks_h5(wd_ds, uniq_cols)
            # Decode (CF-style)
            wsK = (ws_raw / (sf_ws if sf_ws != 0 else 1.0) + of_ws).astype("float32", copy=False)
            wdK = (wd_raw / (sf_wd if sf_wd != 0 else 1.0) + of_wd).astype("float32", copy=False)
            read_s += time.perf_counter() - t0

            t1 = time.perf_counter()
            uK, vK = uv_from_ws_wd(wsK, wdK)  # (T,K)
            uv_s += time.perf_counter() - t1

            t2 = time.perf_counter()
            uS = (W.T.dot(uK.T)).T.astype("float32", copy=False)  # (T,S)
            vS = (W.T.dot(vK.T)).T.astype("float32", copy=False)
            proj_s += time.perf_counter() - t2

            uS_cache[(y, h)] = uS
            vS_cache[(y, h)] = vS

        timing["per_year"][str(y)] = {
            "read_s": round(read_s, 3),
            "uv_s": round(uv_s, 3),
            "project_s": round(proj_s, 3),
            "vertical_s": 0.0,  # will accumulate below
        }

    # 2) For each target height, build Tall x S (concat over years) then quantiles
    all_rows: List[dict] = []
    qnames = [f"q{q:03d}" for q in range(101)]
    q_total = 0.0

    for z in target_heights:
        zf = float(z)
        is_exact, h_lo, h_hi = brackets[int(z)]
        ws_by_year: List[np.ndarray] = []

        for y in YEARS:
            t_vert0 = time.perf_counter()
            if is_exact:
                uS = uS_cache[(y, h_lo)]
                vS = vS_cache[(y, h_lo)]
                wsS = np.hypot(uS, vS, dtype="float32")
            else:
                u_lo = uS_cache[(y, h_lo)]; v_lo = vS_cache[(y, h_lo)]
                u_hi = uS_cache[(y, h_hi)]; v_hi = vS_cache[(y, h_hi)]
                ws_lo = np.hypot(u_lo, v_lo, dtype="float64")
                ws_hi = np.hypot(u_hi, v_hi, dtype="float64")
                bad = (~np.isfinite(ws_lo)) | (~np.isfinite(ws_hi)) | (ws_lo <= 0) | (ws_hi <= 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = np.log(ws_lo/ws_hi) / np.log(float(h_lo)/float(h_hi))
                alpha = np.where(bad | ~np.isfinite(alpha), FALLBACK_ALPHA, alpha)
                factor = (zf / float(h_hi)) ** alpha
                u_z = (u_hi.astype("float64") * factor).astype("float32")
                v_z = (v_hi.astype("float64") * factor).astype("float32")
                wsS = np.hypot(u_z, v_z, dtype="float32")
            timing["per_year"][str(y)]["vertical_s"] = round(
                timing["per_year"][str(y)]["vertical_s"] + (time.perf_counter() - t_vert0), 3
            )
            ws_by_year.append(wsS)

        ws_all = np.vstack(ws_by_year)  # (Tall, S)

        t_q0 = time.perf_counter()
        q = quantile_block(ws_all)      # (101, S)
        q_total += time.perf_counter() - t_q0

        ts = pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"
        for j in range(S):
            row = {
                "grid_id": str(site_ids[j]),
                "lat": float(sub.iloc[j]["lat"]),
                "lon": float(sub.iloc[j]["lon"]),
                "dataset": "WTK CONUS",
                "height_m": zf,
                "interp": f"IDW-{K_NEIGH}-u,v(p={IDW_POWER}); vert={'exact' if is_exact else f'powerlaw(alpha_fallback={FALLBACK_ALPHA})'}",
                "agg": "native_hourly",
                "years": f"{YEARS[0]}-{YEARS[-1]}",
                "processed_utc": ts,
            }
            for qi, qn in enumerate(qnames):
                row[qn] = float(q[qi, j])
            all_rows.append(row)

    timing["quantiles"] = round(q_total, 3)
    timing["compute_total"] = round(time.perf_counter() - t_compute0, 3)

    out = pd.DataFrame(all_rows).sort_values(["grid_id","height_m"]).reset_index(drop=True)
    return out, timing, needed_heights

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="ERA5-grid -> WTK quantiles at ERA5 heights (tile-optimized, IDW-4, power-law), with I/O optimizations.")
    ap.add_argument("--era5-grid", type=Path, required=True,
                    help="CSV/Parquet with columns: grid_id (optional), lat, lon.")
    ap.add_argument("--wtk-dir", type=Path, required=False,
                    help="Directory with local WTK HDF5 files (wtk_conus_2007.h5 ... 2013). Required when processing tiles.")
    ap.add_argument("--heights", type=str, default="30,40,50,60,80,100",
                    help="Comma-separated target heights (m).")
    ap.add_argument("--tile-km", type=float, default=250.0,
                    help="Tile width/height in kilometers (approx).")
    ap.add_argument("--out-dir", type=Path, default=Path("era5grid_wtk_out"),
                    help="Output directory for per-tile files.")
    ap.add_argument("--format", type=str, choices=["parquet","csv"], default="csv",
                    help="Per-tile output format.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite tile outputs if they already exist.")
    # Parity with WTK-LED:
    ap.add_argument("--list-tiles", action="store_true",
                    help="List tile IDs (one per line) to stdout and exit.")
    ap.add_argument("--tile-only", type=str, default=None,
                    help="Process only this tile ID (or comma-separated list of IDs).")

    args = ap.parse_args()

    # Load ERA5 grid (needed for both list-tiles and processing)
    grid = load_era5_grid(args.era5_grid, log_prefix=LOG_PREFIX)

    # Build tiles
    tiles = make_tiles(grid, tile_km=args.tile_km)

    if args.list_tiles:
        for t in sorted(tiles.groups.keys()):
            print(t)
        return

    if not args.wtk_dir:
        raise SystemExit("--wtk-dir is required when processing tiles (omit only with --list-tiles).")

    # Determine target heights
    if args.heights:
        target_heights = np.array(sorted(set(int(round(float(z))) for z in args.heights.split(","))), dtype=int)
    else:
        target_heights = WTK_HEIGHTS.copy()

    log(f"{LOG_PREFIX} ERA5 grid points: {len(grid)}; target heights: {list(target_heights)}")
    log(f"{LOG_PREFIX} Tiling complete: {len(tiles.groups)} tiles at ~{args.tile_km} km")

    # Optional subset of tiles
    tile_items = tiles.groups
    if args.tile_only:
        wanted = set(int(x.strip(), 10) for x in args.tile_only.split(",") if x.strip())
        missing = [t for t in wanted if t not in tile_items]
        if missing:
            log(f"{LOG_PREFIX} Warning: {len(missing)} requested tile(s) not found in grid: {missing}")
        tile_items = {t: idxs for t, idxs in tile_items.items() if t in wanted}
        log(f"{LOG_PREFIX} Processing subset: {len(tile_items)} tile(s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Open WTK year files once; build KDTree from first year
    log(f"{LOG_PREFIX} Opening WTK files for years {YEARS[0]}-{YEARS[-1]} ...")
    year_files: Dict[int, h5py.File] = {y: open_year_file(args.wtk_dir, y) for y in YEARS}

    try:
        lead = year_files[YEARS[0]]
        log(f"{LOG_PREFIX} Building KDTree over WTK coordinates ...")
        tree, idx_map = build_wtk_tree(lead)

        # Process tiles
        for t_id, idxs in tqdm(tile_items.items(), desc="Tiles", unit="tile"):
            t0 = time.perf_counter()

            out_fp = args.out_dir / f"tile_{t_id}.{ 'parquet' if args.format=='parquet' else 'csv'}"
            if out_fp.exists() and not args.overwrite:
                log(f"{LOG_PREFIX} Skip existing {out_fp.name} (use --overwrite to replace).")
                continue

            sub = grid.iloc[idxs].reset_index(drop=True)
            S = len(sub)

            # Neighbor pack
            t_nb0 = time.perf_counter()
            x, y = to_xy_lcc(sub["lon"].to_numpy(dtype="float64"), sub["lat"].to_numpy(dtype="float64"))
            nbh = neighbor_weights_idw4(np.column_stack([x, y]), tree, idx_map, power=IDW_POWER)
            t_nb = time.perf_counter() - t_nb0

            # Compute tile
            t_comp0 = time.perf_counter()
            tile_df, timing, needed_heights = compute_tile_quantiles_optimized(
                tile_idx=idxs,
                era5_df=grid,
                neighbor_pack=nbh,
                year_files=year_files,
                target_heights=target_heights,
            )
            t_comp = time.perf_counter() - t_comp0

            # Write
            t_w0 = time.perf_counter()
            tile_df["grid_id"] = tile_df["grid_id"].astype("string")
            if args.format == "parquet":
                tile_df.to_parquet(out_fp, index=False)
            else:
                tile_df.to_csv(out_fp, index=False)
            t_w = time.perf_counter() - t_w0

            metrics = {
                "tile_id": int(t_id),
                "sites": int(S),
                "k_unique": int(len(nbh.uniq_cols)),
                "years": YEARS,
                "heights": list(map(int, target_heights)),
                "h_needed": list(map(int, needed_heights)),
                "rss_gb": (round(rss_gb(), 3) if rss_gb() is not None else None),
                "timing_s": {
                    "neighbors": round(t_nb, 3),
                    "compute_total": timing["compute_total"],
                    "quantiles": timing["quantiles"],
                    "write": round(t_w, 3),
                    "tile_total": round(t_nb + t_comp + t_w, 3),
                    "per_year": timing["per_year"],
                },
                "out_file": str(out_fp),
                "rows": int(len(tile_df)),
            }
            print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {LOG_PREFIX} TILE_METRICS {json.dumps(metrics)}",
                  file=sys.stderr, flush=True)

            log(f"{LOG_PREFIX} Wrote {len(tile_df)} rows -> {out_fp.name}")

        log(f"{LOG_PREFIX} All tiles complete.")
    finally:
        log(f"{LOG_PREFIX} Closing WTK files...")
        for f in year_files.values():
            try: f.close()
            except Exception: pass
        log(f"{LOG_PREFIX} Done.")

if __name__ == "__main__":
    # Keep BLAS/HDF5 from oversubscribing when you parallelize tiles at shell level
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()
