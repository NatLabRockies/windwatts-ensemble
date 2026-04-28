#!/usr/bin/env python3
"""
ERA5-grid -> HRRR CONUS wind-speed quantiles at ERA5 heights, tile-optimized, with I/O optimizations.

- Spatial:  IDW (K=4) on u,v from HRRR grid -> ERA5 grid points.
- Vertical: Power-law between bracketing HRRR heights (fallback alpha = 1/7).
- Years:    2015-2022; files: <base>/hrrr_nat_f02_conus_YYYY.h5 (ALL heights per year).
- Tiling:   2-D Lambert Conformal tiling to bound memory & I/O.
- Output:   One CSV/Parquet per tile; each row = (grid_id, height_m, q000..q100).

Examples
--------
# List tile IDs (stdout) and exit
python grid_hrrr.py --era5-grid ./era5_grid.csv --list-tiles > tiles.txt

# Process a single tile
python grid_hrrr.py --era5-grid ./era5_grid.csv \\
  --data-dir /datasets/WIND/HRRR --tile-only 123456789 --out-dir era5grid_hrrr_out

# Process all tiles
python grid_hrrr.py --era5-grid ./era5_grid.csv \\
  --data-dir /datasets/WIND/HRRR --out-dir era5grid_hrrr_out --format parquet
"""

from __future__ import annotations
import argparse
import sys, os, time, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
from tqdm.auto import tqdm
from rex.resource_extraction import MultiYearWindX

from wem.utils.logging import log
from wem.utils.spatial import idw_weights_from_dd
from wem.grid._grid_shared import Tiles, NeighborPack, load_era5_grid, make_tiles, contiguous_runs
from wem.utils.wind import uv_from_ws_wd
from wem.utils.quantiles import quantile_block
from wem.utils.power_law import bracket_for_height
from wem.constants import HRRR_HEIGHTS

try:
    import psutil  # optional, for RSS reporting
except Exception:
    psutil = None

# -------------------- Config defaults --------------------
YEARS = list(range(2015, 2023))     # 2015-2022 inclusive
K_NEIGH = 4
IDW_POWER = 1.0
FALLBACK_ALPHA = 1.0 / 7.0
LOG_PREFIX = "[HRRR]"

def rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return None

# -------------------- Neighbors: build weights per tile --------------------
def build_neighbor_pack_for_tile(
    lats: np.ndarray,
    lons: np.ndarray,
    tree: cKDTree,
    power: float = IDW_POWER,
) -> NeighborPack:
    # MultiYearWindX.tree expects (lat, lon) and returns (dist, idx)
    dd, ii = tree.query(np.column_stack([lats, lons]), k=K_NEIGH)  # (S,4)
    dd = dd.astype("float64"); ii = ii.astype("int64")
    S = dd.shape[0]
    w4 = idw_weights_from_dd(dd)  # (S,4)

    # Map unique neighbor columns and build column-sparse CSR (K_unique x S)
    uniq_cols = np.unique(ii.reshape(-1))      # sorted ascending
    pos = {int(c): j for j, c in enumerate(uniq_cols)}

    rows, cols, data = [], [], []
    for s in range(S):
        for n in range(K_NEIGH):
            rows.append(pos[int(ii[s, n])])
            cols.append(s)
            data.append(float(w4[s, n]))
    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(uniq_cols), S), dtype="float64")
    return NeighborPack(uniq_cols=uniq_cols, W=W)

# -------------------- Optimized reads: contiguous blocks --------------------
def read_var_blocks(myr: MultiYearWindX, var: str, uniq_cols_sorted: np.ndarray) -> np.ndarray:
    """
    Read (T, K_unique) for given var using coalesced contiguous blocks.
    Falls back to advanced indexing if slicing fails.
    """
    uniq_cols_sorted = np.asarray(uniq_cols_sorted, dtype=int)
    if uniq_cols_sorted.size == 0:
        return np.zeros((0, 0), dtype="float32")
    runs = contiguous_runs(uniq_cols_sorted)
    blocks: List[np.ndarray] = []
    try:
        for s, e in runs:
            arr = myr[var, :, slice(s, e)]
            blocks.append(np.asarray(arr, dtype="float32"))
        # Concatenate along the column axis
        return np.concatenate(blocks, axis=1).astype("float32", copy=False)
    except Exception:
        # Fallback: advanced indexing (slower)
        arr = myr[var, :, uniq_cols_sorted]
        return np.asarray(arr, dtype="float32")

# -------------------- Core per-tile (optimized) --------------------
def compute_tile_quantiles_optimized(
    tile_idx: np.ndarray,
    era5_df: pd.DataFrame,
    neighbor_pack: NeighborPack,
    myr_by_year: Dict[int, MultiYearWindX],
    target_heights: np.ndarray,
) -> Tuple[pd.DataFrame, dict, List[int]]:
    """
    Optimized per-tile computation for HRRR:
      - For each YEAR: read each needed native HEIGHT once (contiguous column blocks),
        convert to uK/vK and PROJECT once (K->S). Cache uS/vS per (year,height).
      - For each target height z: reuse cached uS/vS to do vertical interpolation (if needed)
        and compute quantiles.
    Returns (DataFrame, timing_dict, needed_heights).
    """
    sub = era5_df.iloc[tile_idx].reset_index(drop=True)
    site_ids = sub["grid_id"].astype("string").to_numpy()
    S = len(sub)
    uniq_cols = neighbor_pack.uniq_cols  # sorted asc
    W = neighbor_pack.W                  # (K_unique, S), csr

    timing = {
        "per_year": {},  # y: {read_s, uv_s, project_s, vertical_s}
        "quantiles": 0.0,
        "compute_total": 0.0,
    }
    t_compute0 = time.perf_counter()

    # Determine native heights needed across all z (avoid duplicate reads)
    needed_heights: List[int] = []
    for z in map(float, target_heights):
        exact, h_lo, h_hi = bracket_for_height(z, HRRR_HEIGHTS)
        if exact:
            needed_heights.append(h_lo)
        else:
            needed_heights.extend([h_lo, h_hi])
    needed_heights = sorted(set(needed_heights))

    # Per-year caches: (year,height) -> uS/vS arrays (T,S)
    uS_cache: Dict[tuple[int,int], np.ndarray] = {}
    vS_cache: Dict[tuple[int,int], np.ndarray] = {}

    # 1) For each year, pre-read all needed heights once
    for y in YEARS:
        myr = myr_by_year.get(y)
        if myr is None:
            continue
        read_s = uv_s = proj_s = 0.0
        for h in needed_heights:
            # Read ws/wd for uniq_cols via contiguous blocks
            t0 = time.perf_counter()
            wsK = read_var_blocks(myr, f"windspeed_{h}m", uniq_cols)   # (T, K)
            wdK = read_var_blocks(myr, f"winddirection_{h}m", uniq_cols)
            read_s += time.perf_counter() - t0

            # Convert to uK/vK
            t1 = time.perf_counter()
            uK, vK = uv_from_ws_wd(wsK, wdK)                           # (T, K)
            uv_s += time.perf_counter() - t1

            # Project K->S once
            t2 = time.perf_counter()
            uS = (W.T.dot(uK.T)).T.astype(np.float32, copy=False)      # (T,S)
            vS = (W.T.dot(vK.T)).T.astype(np.float32, copy=False)
            proj_s += time.perf_counter() - t2

            uS_cache[(y, h)] = uS
            vS_cache[(y, h)] = vS

        timing["per_year"][str(y)] = {
            "read_s": round(read_s, 3),
            "uv_s": round(uv_s, 3),
            "project_s": round(proj_s, 3),
            "vertical_s": 0.0,  # will accumulate below
        }

    all_rows: List[dict] = []
    qnames = [f"q{q:03d}" for q in range(101)]

    # 2) For each target height, build Tall x S (concat over years) then quantiles
    q_total = 0.0
    for z in target_heights:
        zf = float(z)
        is_exact, h_lo, h_hi = bracket_for_height(zf, HRRR_HEIGHTS)

        ws_by_year: List[np.ndarray] = []
        for y in YEARS:
            if (y, h_lo) not in uS_cache and (not is_exact and (y, h_hi) not in uS_cache):
                # This year is missing from cache (file absent); skip
                continue
            t_vert0 = time.perf_counter()
            if is_exact:
                uS = uS_cache[(y, h_lo)]
                vS = vS_cache[(y, h_lo)]
                wsS = np.hypot(uS, vS, dtype="float32")
            else:
                u_lo = uS_cache[(y, h_lo)]
                v_lo = vS_cache[(y, h_lo)]
                u_hi = uS_cache[(y, h_hi)]
                v_hi = vS_cache[(y, h_hi)]

                ws_lo = np.hypot(u_lo, v_lo, dtype="float64")
                ws_hi = np.hypot(u_hi, v_hi, dtype="float64")
                bad = (~np.isfinite(ws_lo)) | (~np.isfinite(ws_hi)) | (ws_lo <= 0) | (ws_hi <= 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = np.log(ws_lo / ws_hi) / np.log(float(h_lo) / float(h_hi))
                alpha = np.where(bad | ~np.isfinite(alpha), FALLBACK_ALPHA, alpha)

                factor = (zf / float(h_hi)) ** alpha
                u_z = (u_hi.astype("float64") * factor).astype("float32")
                v_z = (v_hi.astype("float64") * factor).astype("float32")
                wsS = np.hypot(u_z, v_z, dtype="float32")
            timing["per_year"][str(y)]["vertical_s"] = round(
                timing["per_year"][str(y)]["vertical_s"] + (time.perf_counter() - t_vert0), 3
            )
            ws_by_year.append(wsS)

        if not ws_by_year:
            continue  # no data for this height across any year

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
                "dataset": "HRRR CONUS",
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

# -------------------- File open/cache helpers --------------------
def path_for_year(base_dir: Path, year: int) -> Path:
    # Files: <base>/hrrr_nat_f02_conus_YYYY.h5
    return base_dir / f"hrrr_nat_f02_conus_{year}.h5"

def open_year_cache(base_dir: Path) -> Dict[int, MultiYearWindX]:
    """Open all existing YEAR files; return dict {year: myr}. Require at least one year present."""
    myrs: Dict[int, MultiYearWindX] = {}
    for y in YEARS:
        p = path_for_year(base_dir, y)
        if p.exists():
            myrs[y] = MultiYearWindX(str(p), hsds=False)
        else:
            log(f"{LOG_PREFIX} [WARN] Missing year file: {p} (skipping)")
    if not myrs:
        raise FileNotFoundError(f"No HRRR files found in {base_dir} for years {YEARS[0]}-{YEARS[-1]}")
    return myrs

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="ERA5-grid -> HRRR quantiles at ERA5 heights (tile-optimized, IDW-4, power-law), with I/O optimizations.")
    ap.add_argument("--era5-grid", type=Path, required=True,
                    help="CSV/Parquet with columns: grid_id (optional), lat, lon.")
    ap.add_argument("--data-dir", type=Path, required=False,
                    help="Base folder with per-year HRRR files (hrrr_nat_f02_conus_YYYY.h5). Required when processing tiles.")
    ap.add_argument("--heights", type=str, default="30,40,50,60,80,100",
                    help="Comma-separated target heights (m).")
    ap.add_argument("--tile-km", type=float, default=250.0,
                    help="Tile width/height in kilometers (approx).")
    ap.add_argument("--out-dir", type=Path, default=Path("era5grid_hrrr_out"),
                    help="Output directory for per-tile files.")
    ap.add_argument("--format", type=str, choices=["parquet","csv"], default="csv",
                    help="Per-tile output format.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite tile outputs if they already exist.")
    # Control which tiles to run
    ap.add_argument("--list-tiles", action="store_true",
                    help="List tile IDs (one per line) to stdout and exit.")
    ap.add_argument("--tile-only", type=str, default=None,
                    help="Process only this tile ID (or comma-separated list of IDs).")

    args = ap.parse_args()

    # Load ERA5 grid (needed for both list-tiles and processing)
    grid = load_era5_grid(args.era5_grid, log_prefix=LOG_PREFIX)

    # Build tiles
    tiles = make_tiles(grid, tile_km=args.tile_km)

    # --list-tiles: print to stdout and exit cleanly
    if args.list_tiles:
        for t in sorted(tiles.groups.keys()):
            print(t)
        return

    # From here on, we are processing tiles -> data-dir required
    if not args.data_dir:
        raise SystemExit("--data-dir is required when processing tiles (omit only with --list-tiles).")

    # Determine target heights
    if args.heights:
        target_heights = np.array(sorted(set(int(round(float(z))) for z in args.heights.split(","))), dtype=int)
    else:
        target_heights = np.array([30,40,50,60,80,100], dtype=int)
    log(f"{LOG_PREFIX} ERA5 grid points: {len(grid)}; target heights: {list(target_heights)}")
    log(f"{LOG_PREFIX} Tiling complete: {len(tiles.groups)} tiles at ~{args.tile_km} km")

    # Filter tiles if --tile-only provided
    tile_items = tiles.groups
    if args.tile_only:
        wanted = set(int(x.strip(), 10) for x in args.tile_only.split(",") if x.strip())
        missing = [t for t in wanted if t not in tile_items]
        if missing:
            log(f"{LOG_PREFIX} Warning: {len(missing)} requested tile(s) not found in grid: {missing}")
        tile_items = {t: idxs for t, idxs in tile_items.items() if t in wanted}
        log(f"{LOG_PREFIX} Processing subset: {len(tile_items)} tile(s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build neighbor tree once (use first available year to access .tree)
    myr_by_year = open_year_cache(args.data_dir)
    first_year = sorted(myr_by_year.keys())[0]
    tree: cKDTree = myr_by_year[first_year].tree  # queries with (lat, lon)

    try:
        # Process tiles
        for t_id, idxs in tqdm(tile_items.items(), desc="Tiles", unit="tile"):
            out_fp = args.out_dir / f"tile_{t_id}.{ 'parquet' if args.format=='parquet' else 'csv'}"
            if out_fp.exists() and not args.overwrite:
                log(f"{LOG_PREFIX} Skip existing {out_fp.name} (use --overwrite to replace).")
                continue

            sub = grid.iloc[idxs].reset_index(drop=True)

            # Build neighbors/weights (fast; ~milliseconds)
            t_nb0 = time.perf_counter()
            nbh = build_neighbor_pack_for_tile(
                lats=sub["lat"].to_numpy(dtype="float64"),
                lons=sub["lon"].to_numpy(dtype="float64"),
                tree=tree,
                power=IDW_POWER,
            )
            t_nb = time.perf_counter() - t_nb0

            # Compute tile (optimized path)
            t_comp0 = time.perf_counter()
            tile_df, timing, needed_heights = compute_tile_quantiles_optimized(
                tile_idx=idxs,
                era5_df=grid,
                neighbor_pack=nbh,
                myr_by_year=myr_by_year,
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

            # Structured one-line JSON timing (easy to grep)
            metrics = {
                "tile_id": int(t_id),
                "sites": int(len(sub)),
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

        log(f"{LOG_PREFIX} All tiles complete.")
    finally:
        log(f"{LOG_PREFIX} Closing files...")
        for myr in myr_by_year.values():
            try: myr.close()
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
