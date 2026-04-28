#!/usr/bin/env python3
"""
WTK-LED CONUS (2018-2020) -> quantiles at site-specific heights, using ERA5 gold-standard site list.

- Site list: era5_quantiles_gold_standard_2007_2024.csv (station_id, name, lat, lon, elev_m, height_m)
- Batching: by height (max 100 per batch) -> exact-height read or bracketing heights + power-law vertical interp
- Spatial:  IDW-4 on u,v (nearest 4 grid points from MultiYearWindX.tree)
- Years:    2018-2020
- Output:   wtk_led_conus_quantiles_2018_2020.csv (resume-safe by station_id+height_m)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from rex.resource_extraction import MultiYearWindX

from wem.constants import WTKLED_HEIGHTS
from wem.utils.logging import log
from wem.utils.sites import load_gs_sites, already_done_gs
from wem.utils.wind import uv_from_ws_wd, gather_unique
from wem.utils.quantiles import quantile_block
from wem.utils.spatial import idw_weights_from_dd
from wem.utils.power_law import bracket_for_height

# ──────────────────────────────────────────────────────────────
YEARS          = [2018, 2019, 2020]
NBR_K          = 4
INTERP_SP      = "IDW-4-u,v"
INTERP_V       = "powerlaw"

# ───────────────────── IO helpers ─────────────────────────────
def _read_var(myr: MultiYearWindX, var: str, idxs: np.ndarray) -> np.ndarray:
    """Read a variable for ALL time and selected columns idxs -> returns (T,K) float32."""
    try:
        arr = myr[var, :, idxs]
        return np.asarray(arr, dtype="float32")
    except Exception:
        cols = []
        for i in idxs:
            a = myr[var, :, int(i)]
            cols.append(np.asarray(a, dtype="float32"))
        return np.stack(cols, axis=1).astype("float32")  # (T,K)

# ───────────────────── file open helpers ──────────────────────
def path_for_height_year(base_dir: Path, year: int, h: int) -> Path:
    # Files look like: /datasets/WIND/conus/v2.0.0/2018/conus_2018_10m.h5
    return base_dir / f"{year}" / f"conus_{year}_{h}m.h5"

def open_myr_cache(base_dir: Path, needed: List[tuple[int,int]]) -> Dict[tuple[int,int], MultiYearWindX]:
    """Open all (year,height) myrs once; return dict {(y,h): myr}."""
    myrs: Dict[tuple[int,int], MultiYearWindX] = {}
    for y, h in needed:
        p = path_for_height_year(base_dir, y, h)
        if not p.exists():
            raise FileNotFoundError(p)
        myrs[(y, h)] = MultiYearWindX(str(p), hsds=False)
    return myrs

# ───────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="WTK-LED CONUS -> quantiles at site heights (IDW-4 + power-law), batched by height.")
    ap.add_argument("--sites", type=Path, default=Path("era5_quantiles_gold_standard_2007_2024.csv"),
                    help="CSV with station_id, name, lat, lon, elev_m, height_m (from ERA5 gold-standard run).")
    ap.add_argument("--data-dir", type=Path, default=Path("/datasets/WIND/conus/v2.0.0"),
                    help="Base folder with per-year WTK-LED CONUS files (conus_YYYY_<height>m.h5).")
    ap.add_argument("--out", type=Path, default=Path("wtk_led_conus_quantiles_gold_standard_2018_2020.csv"),
                    help="Output CSV to append rows (resume-safe).")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="Maximum sites per batch (per height).")
    ap.add_argument("--log-every", type=int, default=200,
                    help="Progress log cadence (rows).")
    args = ap.parse_args()

    # Load sites & filter already-done station_id+height_m
    sites_all = load_gs_sites(args.sites)
    done = already_done_gs(args.out)
    mask_done = [(str(s), float(h)) in done for s, h in zip(sites_all["station_id"], sites_all["height_m"])]
    todo = sites_all.loc[~pd.Series(mask_done).values].reset_index(drop=True)
    log(f"[INFO] {len(sites_all)} site-heights total; {len(done)} already in {args.out.name}; {len(todo)} to process.")
    if todo.empty:
        log(f"[INFO] Nothing to do."); return

    # For neighbor queries we just need a tree once; use 2018 10 m file.
    myr_tree = MultiYearWindX(str(path_for_height_year(args.data_dir, YEARS[0], 10)), hsds=False)
    tree = myr_tree.tree

    # Precompute neighbors for ALL to-do rows (weights are independent of height)
    log(f"[INFO] Precomputing nearest-{NBR_K} neighbors for {len(todo)} sites ...")
    dd_all = np.empty((len(todo), NBR_K), dtype="float64")
    ii_all = np.empty((len(todo), NBR_K), dtype="int64")
    for i, (lat, lon) in enumerate(todo[["lat","lon"]].to_numpy(dtype="float64")):
        dd, ii = tree.query((lat, lon), NBR_K)
        dd_all[i] = dd
        ii_all[i] = ii
    w_all = idw_weights_from_dd(dd_all)  # (S,4)

    # Group by target height; process in sub-batches (<= batch-size)
    header_needed = not args.out.exists()
    total_appended = 0
    log(f"[INFO] Processing by height (max batch {args.batch_size}) "
        f"(native cadence; {INTERP_SP}; vert={INTERP_V}) ...")

    for z_val, idxs_height in tqdm(list(todo.groupby("height_m").groups.items()), desc="Heights", unit="height"):
        z = float(z_val)
        idxs_for_height = np.array(sorted(idxs_height), dtype=int)
        exact, h_lo, h_hi = bracket_for_height(z, WTKLED_HEIGHTS)

        # Pre-open required (year,height) files for this height group
        needed_pairs = (
            [(y, h_lo) for y in YEARS] if exact
            else [(y, h_lo) for y in YEARS] + [(y, h_hi) for y in YEARS]
        )

        # remove duplicates
        needed_pairs = sorted(set(needed_pairs))
        myrs = open_myr_cache(args.data_dir, needed_pairs)

        for start in range(0, len(idxs_for_height), args.batch_size):
            sel = idxs_for_height[start:start+args.batch_size]
            batch = todo.iloc[sel].reset_index(drop=True)
            idx4  = ii_all[sel]
            w4    = w_all[sel]
            uniq_idx, pos_map, cols4 = gather_unique(idx4)  # uniq (K,), cols4 (Sb,4)

            ws_years: List[np.ndarray] = []

            for y in YEARS:
                if exact:
                    myr_z = myrs[(y, h_lo)]
                    # Read neighbors (T,K) for both vars in one shot (or loop)
                    wsK = _read_var(myr_z, f"windspeed_{h_lo}m", uniq_idx)  # (T,K)
                    wdK = _read_var(myr_z, f"winddirection_{h_lo}m", uniq_idx)
                    uK, vK = uv_from_ws_wd(wsK, wdK)                       # (T,K)
                    T, K = uK.shape
                    # gather to (T,Sb,4)
                    u_g = uK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                    v_g = vK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                    w = w4[None, :, :]                                     # (1,Sb,4)
                    u_s = (u_g * w).sum(axis=2)                            # (T,Sb)
                    v_s = (v_g * w).sum(axis=2)
                    ws_s = np.sqrt(u_s*u_s + v_s*v_s, dtype="float32")     # (T,Sb)
                    ws_years.append(ws_s)
                else:
                    myr_lo = myrs[(y, h_lo)]
                    myr_hi = myrs[(y, h_hi)]

                    wsK_lo = _read_var(myr_lo, f"windspeed_{h_lo}m", uniq_idx)
                    wdK_lo = _read_var(myr_lo, f"winddirection_{h_lo}m", uniq_idx)
                    wsK_hi = _read_var(myr_hi, f"windspeed_{h_hi}m", uniq_idx)
                    wdK_hi = _read_var(myr_hi, f"winddirection_{h_hi}m", uniq_idx)

                    uK_lo, vK_lo = uv_from_ws_wd(wsK_lo, wdK_lo)
                    uK_hi, vK_hi = uv_from_ws_wd(wsK_hi, wdK_hi)

                    T, K = uK_lo.shape
                    # gather neighbor columns to sites
                    def gather(uK: np.ndarray, vK: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
                        u_g = uK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                        v_g = vK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                        w = w4[None, :, :]
                        u_s = (u_g * w).sum(axis=2)
                        v_s = (v_g * w).sum(axis=2)
                        return u_s, v_s

                    u_lo, v_lo = gather(uK_lo, vK_lo)  # (T,Sb)
                    u_hi, v_hi = gather(uK_hi, vK_hi)

                    ws_lo = np.hypot(u_lo, v_lo, dtype="float64")
                    ws_hi = np.hypot(u_hi, v_hi, dtype="float64")

                    invalid = (~np.isfinite(ws_lo)) | (~np.isfinite(ws_hi)) | (ws_lo <= 0) | (ws_hi <= 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        alpha = np.log(ws_lo/ws_hi) / np.log(float(h_lo)/float(h_hi))  # (T,Sb)
                    alpha = np.where(invalid | ~np.isfinite(alpha), 1.0/7.0, alpha)

                    factor = (z / float(h_hi)) ** alpha
                    u_z = (u_hi.astype("float64") * factor).astype("float32")
                    v_z = (v_hi.astype("float64") * factor).astype("float32")
                    ws_s = np.hypot(u_z, v_z, dtype="float32")
                    ws_years.append(ws_s)

            # Concatenate years and compute quantiles
            ws_all = np.concatenate(ws_years, axis=0)          # (Tall, Sb)
            q = quantile_block(ws_all)                         # (101, Sb)

            # Write batch rows
            cols_q = [f"q{p:03d}" for p in range(101)]
            rows = []
            ts = pd.Timestamp.utcnow().isoformat(timespec="seconds")+"Z"
            for j in range(len(batch)):
                rows.append({
                    "station_id": str(batch.iloc[j]["station_id"]),
                    "name":       str(batch.iloc[j]["name"]),
                    "lat":        float(batch.iloc[j]["lat"]),
                    "lon":        float(batch.iloc[j]["lon"]),
                    "elev_m":     (float(batch.iloc[j]["elev_m"]) if pd.notna(batch.iloc[j]["elev_m"]) else np.nan),
                    "dataset":    "WTK-LED CONUS",
                    "height_m":   float(z),
                    "interp":     f"{INTERP_SP}; vert={INTERP_V if not exact else 'exact'}",
                    "agg":        "native",
                    "years":      "2018-2020",
                    "processed_utc": ts,
                    **{cols_q[qi]: float(q[qi, j]) for qi in range(101)},
                })

            pd.DataFrame(rows).to_csv(args.out, mode="a", header=header_needed, index=False)
            header_needed = False
            total_appended += len(rows)

            if (total_appended % args.log_every == 0) or (total_appended == len(todo)):
                log(f"[INFO] Progress: appended {total_appended}/{len(todo)}")

    log(f"[INFO] Done. Appended {total_appended} row(s) to {args.out}.")

if __name__ == "__main__":
    main()
