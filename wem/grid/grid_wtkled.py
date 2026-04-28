#!/usr/bin/env python3
"""
ERA5-grid -> WTK-LED CONUS wind-speed quantiles at ERA5 heights, tile-optimized, with I/O + compute optimizations.

- Spatial:  IDW (K=4) on u,v from WTK-LED grid -> ERA5 grid points.
- Vertical: Power-law between bracketing WTK-LED heights (fallback alpha = 1/7).
- Years:    2018-2020 (WTK-LED CONUS v2.0.0-style files: conus_YYYY_<height>m.h5).
- Tiling:   2-D Lambert Conformal tiling to bound memory & I/O.
- Output:   One CSV/Parquet per tile; each row = (grid_id, height_m, q000..q100).
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
from wem.constants import WTKLED_HEIGHTS

try:
    import psutil  # optional, for RSS reporting
except Exception:
    psutil = None

# -------------------- Config defaults --------------------
YEARS = [2018, 2019, 2020]
K_NEIGH = 4
IDW_POWER = 1.0
FALLBACK_ALPHA = 1.0 / 7.0
LOG_PREFIX = "[WTKLED]"

def rss_gb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return None

def set_num_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        import mkl  # type: ignore
        mkl.set_num_threads(n)
    except Exception:
        pass
    try:
        import numexpr as ne  # type: ignore
        ne.set_num_threads(n)
    except Exception:
        pass

# -------------------- Neighbors: build weights per tile --------------------
def build_neighbor_pack_for_tile(
    lats: np.ndarray,
    lons: np.ndarray,
    tree: cKDTree,
    power: float = IDW_POWER,
) -> NeighborPack:
    dd, ii = tree.query(np.column_stack([lats, lons]), k=K_NEIGH)  # (S,4)
    dd = dd.astype("float64"); ii = ii.astype("int64")
    S = dd.shape[0]
    w4 = idw_weights_from_dd(dd)  # (S,4)

    uniq_cols = np.unique(ii.reshape(-1))  # sorted ascending
    pos = {int(c): j for j, c in enumerate(uniq_cols)}

    rows, cols, data = [], [], []
    for s in range(S):
        for n in range(K_NEIGH):
            rows.append(pos[int(ii[s, n])]); cols.append(s); data.append(float(w4[s, n]))
    # PARITY: use float64 like HRRR/WTK
    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(uniq_cols), S), dtype="float64")
    return NeighborPack(uniq_cols=uniq_cols, W=W)

# -------------------- Optimized reads: contiguous blocks --------------------
def read_var_blocks(myr: MultiYearWindX, var: str, uniq_cols_sorted: np.ndarray) -> np.ndarray:
    uniq_cols_sorted = np.asarray(uniq_cols_sorted, dtype=int)
    if uniq_cols_sorted.size == 0:
        return np.zeros((0, 0), dtype="float32")
    runs = contiguous_runs(uniq_cols_sorted)
    blocks: List[np.ndarray] = []
    try:
        for s, e in runs:
            arr = myr[var, :, slice(s, e)]
            blocks.append(np.asarray(arr, dtype="float32"))
        return np.concatenate(blocks, axis=1).astype("float32", copy=False)
    except Exception:
        arr = myr[var, :, uniq_cols_sorted]
        return np.asarray(arr, dtype="float32")

# -------------------- Core per-tile (optimized) --------------------
def compute_tile_quantiles_optimized(
    tile_idx: np.ndarray,
    era5_df: pd.DataFrame,
    neighbor_pack: NeighborPack,
    myr_cache: Dict[tuple[int,int], MultiYearWindX],
    target_heights: np.ndarray,
    projection_mode: str,
    W_dense: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, dict, List[int]]:
    sub = era5_df.iloc[tile_idx].reset_index(drop=True)
    site_ids = sub["grid_id"].astype("string").to_numpy()
    S = len(sub)
    uniq_cols = neighbor_pack.uniq_cols
    W = neighbor_pack.W  # CSR (float64)

    timing = {"per_year": {}, "quantiles": 0.0, "compute_total": 0.0}
    t_compute0 = time.perf_counter()

    # Determine native heights needed across all z (avoid duplicate reads)
    needed_heights: List[int] = []
    for z in map(float, target_heights):
        exact, h_lo, h_hi = bracket_for_height(z, WTKLED_HEIGHTS)
        needed_heights.extend([h_lo] if exact else [h_lo, h_hi])
    needed_heights = sorted(set(needed_heights))

    # Per-year caches: (year,height) -> uS/vS arrays (T,S)
    uS_cache: Dict[tuple[int,int], np.ndarray] = {}
    vS_cache: Dict[tuple[int,int], np.ndarray] = {}

    for y in YEARS:
        read_s = uv_s = proj_s = 0.0
        for h in needed_heights:
            myr = myr_cache[(y, h)]
            # Read ws/wd for uniq_cols via contiguous blocks
            t0 = time.perf_counter()
            wsK = read_var_blocks(myr, f"windspeed_{h}m", uniq_cols)   # (T, K) float32
            wdK = read_var_blocks(myr, f"winddirection_{h}m", uniq_cols)
            read_s += time.perf_counter() - t0

            # Convert to uK/vK
            t1 = time.perf_counter()
            uK, vK = uv_from_ws_wd(wsK, wdK)                           # (T, K) float32
            uv_s += time.perf_counter() - t1

            # Project K->S
            t2 = time.perf_counter()
            if projection_mode == "dense" and W_dense is not None:
                # PARITY: keep W_dense float64 so result is float64, then cast like HRRR/WTK
                uS = (uK @ W_dense).astype("float32", copy=False)
                vS = (vK @ W_dense).astype("float32", copy=False)
            else:
                uS = (W.T.dot(uK.T)).T.astype("float32", copy=False)   # W is float64 -> float64 -> cast
                vS = (W.T.dot(vK.T)).T.astype("float32", copy=False)
            proj_s += time.perf_counter() - t2

            uS_cache[(y, h)] = uS
            vS_cache[(y, h)] = vS

        timing["per_year"][str(y)] = {
            "read_s": round(read_s, 3),
            "uv_s": round(uv_s, 3),
            "project_s": round(proj_s, 3),
            "vertical_s": 0.0,
        }

    all_rows: List[dict] = []
    qnames = [f"q{q:03d}" for q in range(101)]
    q_total = 0.0

    for z in target_heights:
        zf = float(z)
        is_exact, h_lo, h_hi = bracket_for_height(zf, WTKLED_HEIGHTS)

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

                # PARITY: compute ws_lo/ws_hi, alpha, factor in float64 (match HRRR/WTK)
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

        ws_all = np.vstack(ws_by_year)  # (Tall, S) float32

        t_q0 = time.perf_counter()
        q = quantile_block(ws_all)      # (101, S) float32
        q_total += time.perf_counter() - t_q0

        ts = pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"
        for j in range(S):
            row = {
                "grid_id": str(site_ids[j]),
                "lat": float(sub.iloc[j]["lat"]),
                "lon": float(sub.iloc[j]["lon"]),
                "dataset": "WTK-LED CONUS",
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
def path_for_height_year(base_dir: Path, year: int, h: int) -> Path:
    return base_dir / f"{year}" / f"conus_{year}_{h}m.h5"

def open_myr_cache(base_dir: Path, target_heights: np.ndarray) -> Dict[tuple[int,int], MultiYearWindX]:
    needed: set[tuple[int,int]] = set()
    for z in target_heights:
        exact, h_lo, h_hi = bracket_for_height(float(z), WTKLED_HEIGHTS)
        if exact:
            for y in YEARS: needed.add((y, h_lo))
        else:
            for y in YEARS: needed.update({(y, h_lo), (y, h_hi)})
    myrs: Dict[tuple[int,int], MultiYearWindX] = {}
    for (y, h) in sorted(needed):
        p = path_for_height_year(base_dir, y, h)
        if not p.exists():
            raise FileNotFoundError(p)
        myrs[(y, h)] = MultiYearWindX(str(p), hsds=False)
    return myrs

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="ERA5-grid -> WTK-LED quantiles at ERA5 heights (tile-optimized, IDW-4, power-law), with I/O+compute parity with HRRR/WTK.")
    ap.add_argument("--era5-grid", type=Path, required=True,
                    help="CSV/Parquet with columns: grid_id (optional), lat, lon.")
    ap.add_argument("--data-dir", type=Path, required=False,
                    help="Base folder with WTK-LED per-year files (e.g., /datasets/WIND/conus/v2.0.0). Required when processing tiles.")
    ap.add_argument("--heights", type=str, default="30,40,50,60,80,100",
                    help="Comma-separated target heights (m).")
    ap.add_argument("--tile-km", type=float, default=250.0,
                    help="Tile width/height in kilometers (approx).")
    ap.add_argument("--out-dir", type=Path, default=Path("era5grid_wtkled_conus_out"),
                    help="Output directory for per-tile files.")
    ap.add_argument("--format", type=str, choices=["parquet","csv"], default="csv",
                    help="Per-tile output format.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite tile outputs if they already exist.")
    ap.add_argument("--list-tiles", action="store_true",
                    help="List tile IDs (one per line) to stdout and exit.")
    ap.add_argument("--tile-only", type=str, default=None,
                    help="Process only this tile ID (or comma-separated list of IDs).")
    # PARITY: default to sparse to match HRRR/WTK exactly
    ap.add_argument("--projection", type=str, choices=["auto","sparse","dense"], default="sparse",
                    help="K->S projection implementation per tile. 'sparse' matches HRRR/WTK bitwise better.")
    ap.add_argument("--dense-max-mb", type=float, default=64.0,
                    help="Max MB allowed to densify W per tile when --projection=auto.")
    ap.add_argument("--num-threads", type=int, default=1,
                    help="Threads per process to use for BLAS/VM/NumExpr.")

    args = ap.parse_args()

    set_num_threads(max(1, int(args.num_threads)))

    grid = load_era5_grid(args.era5_grid, log_prefix=LOG_PREFIX)
    tiles = make_tiles(grid, tile_km=args.tile_km)

    if args.list_tiles:
        for t in sorted(tiles.groups.keys()):
            print(t)
        return

    if not args.data_dir:
        raise SystemExit("--data-dir is required when processing tiles (omit only with --list-tiles).")

    if args.heights:
        target_heights = np.array(sorted(set(int(round(float(z))) for z in args.heights.split(","))), dtype=int)
    else:
        target_heights = np.array([30,40,50,60,80,100], dtype=int)

    log(f"{LOG_PREFIX} ERA5 grid points: {len(grid)}; target heights: {list(target_heights)}")
    log(f"{LOG_PREFIX} Tiling complete: {len(tiles.groups)} tiles at ~{args.tile_km} km")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Neighbor tree from one file (10 m, 2018)
    lead = MultiYearWindX(str(path_for_height_year(args.data_dir, YEARS[0], 10)), hsds=False)
    tree: cKDTree = lead.tree

    myr_cache = open_myr_cache(args.data_dir, target_heights)

    try:
        for t_id, idxs in tqdm(tiles.groups.items(), desc="Tiles", unit="tile"):
            if args.tile_only:
                wanted = set(int(x.strip(), 10) for x in args.tile_only.split(",") if x.strip())
                if t_id not in wanted:
                    continue

            out_fp = args.out_dir / f"tile_{t_id}.{ 'parquet' if args.format=='parquet' else 'csv'}"
            if out_fp.exists() and not args.overwrite:
                log(f"{LOG_PREFIX} Skip existing {out_fp.name} (use --overwrite to replace).")
                continue

            sub = grid.iloc[idxs].reset_index(drop=True)

            # Neighbors/weights (float64 W to match HRRR/WTK)
            t_nb0 = time.perf_counter()
            nbh = build_neighbor_pack_for_tile(
                lats=sub["lat"].to_numpy(dtype="float64"),
                lons=sub["lon"].to_numpy(dtype="float64"),
                tree=tree,
                power=IDW_POWER,
            )
            t_nb = time.perf_counter() - t_nb0

            # Decide projection mode
            K = int(len(nbh.uniq_cols)); S = int(len(sub))
            bytes_dense = K * S * 8  # PARITY: float64 for W_dense
            mb_dense = bytes_dense / (1024**2)
            if args.projection == "dense" or (args.projection == "auto" and mb_dense <= args.dense_max_mb):
                proj_mode = "dense"
                W_dense = nbh.W.toarray()  # keep float64
            else:
                proj_mode = "sparse"
                W_dense = None

            # Compute tile
            t_comp0 = time.perf_counter()
            tile_df, timing, needed_heights = compute_tile_quantiles_optimized(
                tile_idx=idxs,
                era5_df=grid,
                neighbor_pack=nbh,
                myr_cache=myr_cache,
                target_heights=target_heights,
                projection_mode=proj_mode,
                W_dense=W_dense,
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
                "sites": int(len(sub)),
                "k_unique": int(len(nbh.uniq_cols)),
                "years": YEARS,
                "heights": list(map(int, target_heights)),
                "h_needed": list(map(int, needed_heights)),
                "rss_gb": (round(rss_gb(), 3) if rss_gb() is not None else None),
                "projection": proj_mode,
                "threads": int(args.num_threads),
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
        try: lead.close()
        except Exception: pass
        for myr in myr_cache.values():
            try: myr.close()
            except Exception: pass
        log(f"{LOG_PREFIX} Done.")

if __name__ == "__main__":
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()
