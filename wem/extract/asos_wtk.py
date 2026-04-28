#!/usr/bin/env python3
"""
WTK (local HDF5) wind-speed quantiles at 10 m for many sites -- batched & scaled.

Fix: apply per-variable scale_factor/add_offset from HDF5 so outputs are in m/s.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from wem.utils.logging import log
from wem.utils.columns import choose_col
from wem.utils.sites import load_sites, already_done
from wem.utils.wind import uv_from_ws_wd
from wem.utils.quantiles import quantile_block
from wem.utils.spatial import to_xy_lcc

# ──────────────────────────────────────────────────────────────
HEIGHT_M = 10
YEARS    = list(range(2007, 2014))  # inclusive 2007-2013
INTERP   = "IDW-4-u,v"

# ─────────────── coordinates + KDTree (simplified) ────────────
def open_year_file(data_dir: Path, year: int) -> h5py.File:
    for name in (f"wtk_conus_{year}.h5", f"conus_{year}.h5", f"conus-{year}.h5"):
        fp = data_dir / name
        if fp.exists():
            return h5py.File(fp, "r")
    raise FileNotFoundError(f"No local WTK file found for year {year} in {data_dir}")

def discover_tree_and_index(lead: h5py.File) -> tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(lead["coordinates"])  # (N,2) = [lat, lon]
    lat = coords[:, 0].astype("float64")
    lon = coords[:, 1].astype("float64")
    log(f"[INFO] coordinates (lat): min={np.nanmin(lat):.4f}, max={np.nanmax(lat):.4f}")
    log(f"[INFO] coordinates (lon): min={np.nanmin(lon):.4f}, max={np.nanmax(lon):.4f}")
    x, y = to_xy_lcc(lon, lat)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.all(ok):
        log(f"[INFO] Dropping {np.size(ok) - int(ok.sum())} non-finite points from KDTree")
    x = x[ok]; y = y[ok]
    idx_map = np.nonzero(ok)[0].astype(np.int64)  # kd-tree idx -> column idx
    tree = cKDTree(np.column_stack([x, y]))
    return tree, idx_map, lat[ok], lon[ok]

# ───────────────────── IO helpers ─────────────────────────────
def get_scale(ds: h5py.Dataset) -> tuple[float, float]:
    """Read scale_factor/add_offset; default to 1.0/0.0."""
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

# ───────────────────────── batching ───────────────────────────
def precompute_neighbors(
    sites: pd.DataFrame, tree: cKDTree, idx_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    sx, sy = to_xy_lcc(sites["lon"].to_numpy(), sites["lat"].to_numpy())
    dist, idxs_tree = tree.query(np.column_stack([sx, sy]), k=4)   # (S,4)
    idxs_tree = idxs_tree.astype(int)
    dist = dist.astype(float)

    # Build weights (IDW). If distance==0 -> one-hot on that neighbor.
    w = np.empty_like(dist, dtype="float64")
    for i in range(dist.shape[0]):
        d = dist[i]
        if np.any(d == 0.0):
            w[i] = 0.0
            w[i, np.argmin(d)] = 1.0
        else:
            wi = 1.0 / d
            w[i] = wi / wi.sum()
    idxs_flat = idx_map[idxs_tree]   # (S,4) map kd-tree->column
    return idxs_flat, w

def build_weight_matrix(
    batch_idxs: np.ndarray, batch_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Dict[int,int]]:
    """
    From (S,4) neighbor columns and (S,4) weights, build:
      - uniq_cols: sorted unique columns
      - W (K x S) dense matrix so that [T,K] @ W = [T,S]
      - col_to_pos mapping
    """
    S = batch_idxs.shape[0]
    uniq_cols = np.unique(batch_idxs.ravel())
    col_to_pos = {int(c): j for j, c in enumerate(uniq_cols)}
    K = len(uniq_cols)
    W = np.zeros((K, S), dtype="float64")
    for s in range(S):
        for n in range(4):
            c = int(batch_idxs[s, n])
            W[col_to_pos[c], s] += float(batch_w[s, n])
    return uniq_cols, W, col_to_pos

# ─────────────────────── main pipeline ────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Fast WTK -> quantiles @10 m (batched, scaled).")
    ap.add_argument("--sites", type=Path, default=Path("era5_quantiles_2007_2024.csv"),
                    help="CSV with station_id, name, lat, lon, elev_m (ERA5 quantiles file works).")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Folder with local WTK files (wtk_conus_2007.h5 ... 2013).")
    ap.add_argument("--out", type=Path, default=Path("wtk_quantiles_2007_2013.csv"),
                    help="Output CSV to append rows.")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="Sites per batch (increase for speed if RAM allows).")
    ap.add_argument("--log-every", type=int, default=200,
                    help="Progress log cadence (sites).")
    args = ap.parse_args()

    # Load sites & filter ones already done
    sites = load_sites(args.sites)
    done = already_done(args.out)
    todo = sites[~sites["station_id"].isin(done)].reset_index(drop=True)
    log(f"[INFO] {len(sites)} sites total; {len(done)} already in {args.out.name}; {len(todo)} to process.")

    # Open year files
    log(f"[INFO] Opening {len(YEARS)} WTK files (2007-2013)...")
    year_files: Dict[int, h5py.File] = {y: open_year_file(args.data_dir, y) for y in YEARS}

    try:
        # Build KDTree once
        lead = year_files[YEARS[0]]
        log(f"[INFO] Building KDTree from coordinates ...")
        tree, idx_map, _, _ = discover_tree_and_index(lead)

        # Inspect scales once (log)
        ws_ds = lead["windspeed_10m"]
        wd_ds = lead["winddirection_10m"]
        ws_sf, ws_ofs = get_scale(ws_ds)
        wd_sf, wd_ofs = get_scale(wd_ds)
        log(f"[INFO] scale windspeed_10m: scale_factor={ws_sf} add_offset={ws_ofs}")
        log(f"[INFO] scale winddirection_10m: scale_factor={wd_sf} add_offset={wd_ofs}")

        # Precompute neighbors & weights for all todo sites (once)
        log(f"[INFO] Precomputing neighbors/weights for {len(todo)} sites ...")
        nbr_idxs_all, w_all = precompute_neighbors(todo, tree, idx_map)

        header_needed = not args.out.exists()
        total_appended = 0

        log(f"[INFO] Processing {len(todo)} sites in batches of {args.batch_size} "
            f"(native hourly; {INTERP}) ...")
        with tqdm(total=len(todo), unit="site") as pbar:
            for start in range(0, len(todo), args.batch_size):
                end = min(start + args.batch_size, len(todo))
                batch = todo.iloc[start:end].reset_index(drop=True)
                b_idxs = nbr_idxs_all[start:end]
                b_w    = w_all[start:end]
                S = len(batch)

                uniq_cols, W, col_to_pos = build_weight_matrix(b_idxs, b_w)
                K = len(uniq_cols)

                series_list: List[np.ndarray] = []
                for y in YEARS:
                    f = year_files[y]
                    ws = f["windspeed_10m"]      # (T, N)
                    wd = f["winddirection_10m"]  # (T, N)

                    # One read per var for all needed columns (raw -> scale)
                    ws_raw = np.asarray(ws[:, uniq_cols])
                    wd_raw = np.asarray(wd[:, uniq_cols])

                    # Apply scale/offset to physical units
                    wsK = (ws_raw.astype("float32") / ws_sf + ws_ofs)   # m/s
                    wdK = (wd_raw.astype("float32") / wd_sf + wd_ofs)   # degrees (met)

                    uK, vK = uv_from_ws_wd(wsK, wdK)         # (T,K), (T,K)
                    uS = uK @ W                       # (T,S)
                    vS = vK @ W
                    spdS = np.hypot(uS, vS).astype("float32")  # (T,S)
                    series_list.append(spdS)

                spd_all = np.vstack(series_list)      # (Tall, S)
                q_block = quantile_block(spd_all)     # (101, S)

                rows = []
                qs = [f"q{q:03d}" for q in range(101)]
                for j in range(S):
                    row = {
                        "station_id": str(batch.iloc[j]["station_id"]),
                        "name":       str(batch.iloc[j]["name"]),
                        "lat":        float(batch.iloc[j]["lat"]),
                        "lon":        float(batch.iloc[j]["lon"]),
                        "elev_m":     (float(batch.iloc[j]["elev_m"]) if pd.notna(batch.iloc[j]["elev_m"]) else np.nan),
                        "dataset":    "WTK",
                        "height_m":   HEIGHT_M,
                        "interp":     INTERP,
                        "agg":        "native_hourly",
                        "years":      f"{YEARS[0]}-{YEARS[-1]}",
                        "processed_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds")+"Z",
                    }
                    for qi, qname in enumerate(qs):
                        row[qname] = float(q_block[qi, j])
                    rows.append(row)

                pd.DataFrame(rows).to_csv(args.out, mode="a", header=header_needed, index=False)
                header_needed = False
                total_appended += S
                pbar.update(S)

                if total_appended % args.log_every == 0:
                    tqdm.write(f"[INFO] Progress: appended {total_appended}/{len(todo)}")

        log(f"[INFO] Done. Appended {total_appended} row(s) to {args.out}.")
    finally:
        log(f"[INFO] Closing files...")
        for f in year_files.values():
            try: f.close()
            except Exception: pass
        log(f"[INFO] Complete.")

if __name__ == "__main__":
    main()
