#!/usr/bin/env python3
"""
WTK (local HDF5) wind-speed quantiles at site-specific heights for many sites.

Key changes:
- Site list now comes from ERA5 results: era5_quantiles_gold_standard_2007_2024.csv
  (ensures we use the exact set of sites that have ERA5 coverage).
- Batching is done by height value, with a maximum batch size of 100 so each
  batch shares the same target height; we read either that exact WTK height or
  its bracketing pair, minimizing I/O.

Spatial:  IDW-4 on u,v from 4 nearest grid points.
Vertical: Power law between nearest WTK heights (fallback alpha = 1/7).
Years:    2007-2013 inclusive.

Outputs append to wtk_quantiles_2007_2013.csv (resume-safe by station_id+height_m).
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

from wem.constants import WTK_HEIGHTS
from wem.utils.logging import log
from wem.utils.sites import load_gs_sites, already_done_gs
from wem.utils.wind import uv_from_ws_wd
from wem.utils.quantiles import quantile_block
from wem.utils.spatial import to_xy_lcc
from wem.utils.power_law import bracket_for_height

# ──────────────────────────────────────────────────────────────
YEARS      = list(range(2007, 2014))  # inclusive 2007-2013
NBR_K      = 4
INTERP_SP  = "IDW-4-u,v"
INTERP_V   = "powerlaw"

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

# ───────────────────────── neighbors ──────────────────────────
def precompute_neighbors(
    sites: pd.DataFrame, tree: cKDTree, idx_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nbr_cols, weights) each shape (S,4) for all site rows."""
    sx, sy = to_xy_lcc(sites["lon"].to_numpy(), sites["lat"].to_numpy())
    dist, idxs_tree = tree.query(np.column_stack([sx, sy]), k=NBR_K)   # (S,4)
    idxs_tree = idxs_tree.astype(int)
    dist = dist.astype(float)

    # Build weights (IDW). If any distance==0 -> one-hot on that neighbor.
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
        for n in range(NBR_K):
            c = int(batch_idxs[s, n])
            W[col_to_pos[c], s] += float(batch_w[s, n])
    return uniq_cols, W, col_to_pos

# ─────────────────────── main pipeline ───────────────────────
def main():
    ap = argparse.ArgumentParser(description="WTK -> quantiles at site heights (IDW-4 + power-law vertical), batched by height.")
    ap.add_argument("--sites", type=Path, default=Path("era5_quantiles_gold_standard_2007_2024.csv"),
                    help="CSV with station_id, name, lat, lon, elev_m, height_m (from ERA5 gold-standard run).")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Folder with local WTK files (wtk_conus_2007.h5 ... 2013).")
    ap.add_argument("--out", type=Path, default=Path("wtk_quantiles_gold_standard_2007_2013.csv"),
                    help="Output CSV to append rows.")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="Maximum sites per batch (per height).")
    ap.add_argument("--log-every", type=int, default=200,
                    help="Progress log cadence (rows).")
    args = ap.parse_args()

    # Load sites & filter ones already done (by station_id + height_m)
    sites_all = load_gs_sites(args.sites)
    done = already_done_gs(args.out)
    mask_done = [(str(s), float(h)) in done for s, h in zip(sites_all["station_id"], sites_all["height_m"])]
    todo = sites_all.loc[~pd.Series(mask_done).values].reset_index(drop=True)
    log(f"[INFO] {len(sites_all)} site-heights total; {len(done)} already in {args.out.name}; {len(todo)} to process.")

    if todo.empty:
        log(f"[INFO] Nothing to do.")
        return

    # Open year files
    log(f"[INFO] Opening {len(YEARS)} WTK files (2007-2013)...")
    year_files: Dict[int, h5py.File] = {y: open_year_file(args.data_dir, y) for y in YEARS}

    try:
        # Build KDTree once
        lead = year_files[YEARS[0]]
        log(f"[INFO] Building KDTree from coordinates ...")
        tree, idx_map, _, _ = discover_tree_and_index(lead)

        # Precompute neighbors & weights for ALL todo rows once
        log(f"[INFO] Precomputing neighbors/weights for {len(todo)} site-rows ...")
        nbr_cols_all, w_all = precompute_neighbors(todo, tree, idx_map)

        header_needed = not args.out.exists()
        total_appended = 0

        # Process groups by height_m -> batches up to args.batch_size each
        log(f"[INFO] Processing by height (max batch {args.batch_size}) "
            f"(native hourly; {INTERP_SP}; vert={INTERP_V}) ...")

        for z_val, group_idx in tqdm(list(todo.groupby("height_m").groups.items()), desc="Heights", unit="height"):
            z = float(z_val)
            idxs_for_height = np.array(sorted(group_idx), dtype=int)
            exact, h_lo, h_hi = bracket_for_height(z, WTK_HEIGHTS)

            # Process this height group in chunks of <= batch_size
            for start in range(0, len(idxs_for_height), args.batch_size):
                sel = idxs_for_height[start:start+args.batch_size]
                batch = todo.iloc[sel].reset_index(drop=True)

                # Slice neighbor columns/weights for these rows
                b_idxs = nbr_cols_all[sel]
                b_w    = w_all[sel]
                S = len(batch)

                uniq_cols, W, col_to_pos = build_weight_matrix(b_idxs, b_w)
                K = len(uniq_cols)

                series_list: List[np.ndarray] = []

                for y in YEARS:
                    f = year_files[y]

                    # Always read the "upper" and "lower" we need for this height (or the exact only)
                    heights_to_read = [h_lo] if exact else [h_lo, h_hi]
                    uK_by_h: Dict[int, np.ndarray] = {}
                    vK_by_h: Dict[int, np.ndarray] = {}

                    for h in heights_to_read:
                        ws_ds = f[f"windspeed_{h}m"]
                        wd_ds = f[f"winddirection_{h}m"]
                        sf_ws, of_ws = get_scale(ws_ds)
                        sf_wd, of_wd = get_scale(wd_ds)

                        ws_raw = np.asarray(ws_ds[:, uniq_cols])  # (T,K)
                        wd_raw = np.asarray(wd_ds[:, uniq_cols])  # (T,K)

                        wsK = (ws_raw.astype("float32") / (sf_ws if sf_ws != 0 else 1.0) + of_ws)
                        wdK = (wd_raw.astype("float32") / (sf_wd if sf_wd != 0 else 1.0) + of_wd)

                        uK, vK = uv_from_ws_wd(wsK, wdK)  # (T,K)
                        uK_by_h[int(h)] = uK
                        vK_by_h[int(h)] = vK

                    # Project neighbors to sites: uS_h = uK_h @ W
                    uS_by_h: Dict[int, np.ndarray] = {}
                    vS_by_h: Dict[int, np.ndarray] = {}
                    for h in heights_to_read:
                        uS_by_h[int(h)] = uK_by_h[int(h)] @ W  # (T,S)
                        vS_by_h[int(h)] = vK_by_h[int(h)] @ W  # (T,S)

                    # Build site-height time series at target z
                    T = next(iter(uS_by_h.values())).shape[0]
                    ws_batch = np.empty((T, S), dtype="float32")

                    if exact:
                        u_z = uS_by_h[h_lo]
                        v_z = vS_by_h[h_lo]
                        ws_batch[:] = np.hypot(u_z, v_z, dtype="float32")
                    else:
                        u_lo = uS_by_h[h_lo]
                        v_lo = vS_by_h[h_lo]
                        u_hi = uS_by_h[h_hi]
                        v_hi = vS_by_h[h_hi]

                        ws_lo = np.hypot(u_lo, v_lo, dtype="float64")
                        ws_hi = np.hypot(u_hi, v_hi, dtype="float64")

                        invalid = (~np.isfinite(ws_lo)) | (~np.isfinite(ws_hi)) | (ws_lo <= 0) | (ws_hi <= 0)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            alpha = np.log(ws_lo/ws_hi) / np.log(float(h_lo)/float(h_hi))  # (T,S)
                        alpha = np.where(invalid | ~np.isfinite(alpha), 1.0/7.0, alpha)

                        factor = (z / float(h_hi)) ** alpha  # (T,S)
                        u_z = (u_hi.astype("float64") * factor).astype("float32")
                        v_z = (v_hi.astype("float64") * factor).astype("float32")

                        ws_batch[:] = np.hypot(u_z, v_z, dtype="float32")

                    series_list.append(ws_batch)

                # Concatenate years and compute quantiles per site in batch
                ws_all = np.vstack(series_list)            # (Tall, S)
                q_block = quantile_block(ws_all)           # (101, S)

                # Write rows
                rows = []
                qs = [f"q{q:03d}" for q in range(101)]
                nowz = pd.Timestamp.utcnow().isoformat(timespec="seconds")+"Z"
                for j in range(S):
                    row = {
                        "station_id": str(batch.iloc[j]["station_id"]),
                        "name":       str(batch.iloc[j]["name"]),
                        "lat":        float(batch.iloc[j]["lat"]),
                        "lon":        float(batch.iloc[j]["lon"]),
                        "elev_m":     (float(batch.iloc[j]["elev_m"]) if pd.notna(batch.iloc[j]["elev_m"]) else np.nan),
                        "dataset":    "WTK",
                        "height_m":   float(z),
                        "interp":     f"{INTERP_SP}; vert={INTERP_V}",
                        "agg":        "native_hourly",
                        "years":      f"{YEARS[0]}-{YEARS[-1]}",
                        "processed_utc": nowz,
                    }
                    for qi, qname in enumerate(qs):
                        row[qname] = float(q_block[qi, j])
                    rows.append(row)

                pd.DataFrame(rows).to_csv(args.out, mode="a", header=header_needed, index=False)
                header_needed = False
                total_appended += S

                if total_appended % args.log_every == 0 or total_appended == len(todo):
                    log(f"[INFO] Progress: appended {total_appended}/{len(todo)}")

        log(f"[INFO] Done. Appended {total_appended} row(s) to {args.out}.")
    finally:
        log(f"[INFO] Closing files...")
        for f in year_files.values():
            try: f.close()
            except Exception: pass
        log(f"[INFO] Complete.")

if __name__ == "__main__":
    main()
