#!/usr/bin/env python3
"""
HRRR CONUS (2015-2022) -> quantiles at site-specific heights, using ERA5 gold-standard site list.

- Site list: era5_quantiles_gold_standard_2007_2024.csv (station_id, name, lat, lon, elev_m, height_m)
- Batching: by height (max 100 per batch) -> exact-height read or bracketing heights + power-law vertical interp
- Spatial:  IDW-4 on u,v (nearest 4 grid points from MultiYearWindX.tree)
- Years:    2015-2022
- Files:    /datasets/WIND/HRRR/hrrr_nat_f02_conus_YYYY.h5  (each year contains ALL heights)
- Output:   hrrr_quantiles_gold_standard_2015_2022.csv (resume-safe by station_id+height_m)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from rex.resource_extraction import MultiYearWindX

from wem.constants import HRRR_HEIGHTS
from wem.utils.logging import log
from wem.utils.sites import load_gs_sites, already_done_gs
from wem.utils.wind import uv_from_ws_wd, gather_unique
from wem.utils.quantiles import quantile_block
from wem.utils.spatial import idw_weights_from_dd
from wem.utils.power_law import bracket_for_height

# ──────────────────────────────────────────────────────────────
# Available native heights in HRRR files (adjust if needed)
YEARS        = list(range(2015, 2023))  # 2015-2022 inclusive
NBR_K        = 4
INTERP_SP    = "IDW-4-u,v"
INTERP_V     = "powerlaw"

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
def path_for_year(base_dir: Path, year: int) -> Path:
    # Files: /datasets/WIND/HRRR/hrrr_nat_f02_conus_YYYY.h5
    return base_dir / f"hrrr_nat_f02_conus_{year}.h5"

def open_year_cache(base_dir: Path) -> Dict[int, MultiYearWindX]:
    """Open all existing YEAR files; return dict {year: myr}. Require at least one year."""
    myrs: Dict[int, MultiYearWindX] = {}
    for y in YEARS:
        p = path_for_year(base_dir, y)
        if p.exists():
            myrs[y] = MultiYearWindX(str(p), hsds=False)
        else:
            log(f"[INFO] [WARN] Missing year file: {p} (skipping)")
    if not myrs:
        raise FileNotFoundError(f"No HRRR files found in {base_dir} for years {YEARS[0]}-{YEARS[-1]}")
    return myrs

# ───────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="HRRR CONUS -> quantiles at site heights (IDW-4 + power-law), batched by height.")
    ap.add_argument("--sites", type=Path, default=Path("era5_quantiles_gold_standard_2007_2024.csv"),
                    help="CSV with station_id, name, lat, lon, elev_m, height_m (from ERA5 gold-standard run).")
    ap.add_argument("--data-dir", type=Path, default=Path("/datasets/WIND/HRRR"),
                    help="Base folder with per-year HRRR files (hrrr_nat_f02_conus_YYYY.h5).")
    ap.add_argument("--out", type=Path, default=Path("hrrr_quantiles_gold_standard_2015_2022.csv"),
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

    # Open available year files once; build KDTree from first available year
    myrs_by_year = open_year_cache(args.data_dir)
    first_year = sorted(myrs_by_year.keys())[0]
    tree = myrs_by_year[first_year].tree

    # Precompute neighbors for ALL to-do rows (weights are independent of height)
    log(f"[INFO] Precomputing nearest-{NBR_K} neighbors for {len(todo)} sites ...")
    dd_all = np.empty((len(todo), NBR_K), dtype="float64")
    ii_all = np.empty((len(todo), NBR_K), dtype="int64")
    for i, (lat, lon) in enumerate(todo[["lat","lon"]].to_numpy(dtype="float64")):
        dd, ii = tree.query((lat, lon), NBR_K)
        dd_all[i] = dd
        ii_all[i] = ii
    w_all = idw_weights_from_dd(dd_all)  # (S,4)

    # Available native heights (static list above); if needed, you can refine this by probing variables.
    avail = np.sort(HRRR_HEIGHTS)

    # Group by target height; process in sub-batches (<= batch-size)
    header_needed = not args.out.exists()
    total_appended = 0
    log(f"[INFO] Processing by height (max batch {args.batch_size}) "
        f"(native cadence; {INTERP_SP}; vert={INTERP_V}) ...")

    for z_val, idxs_height in tqdm(list(todo.groupby("height_m").groups.items()), desc="Heights", unit="height"):
        z = float(z_val)
        idxs_for_height = np.array(sorted(idxs_height), dtype=int)
        exact, h_lo, h_hi = bracket_for_height(z, avail)

        for start in range(0, len(idxs_for_height), args.batch_size):
            sel = idxs_for_height[start:start+args.batch_size]
            batch = todo.iloc[sel].reset_index(drop=True)
            idx4  = ii_all[sel]
            w4    = w_all[sel]
            uniq_idx, pos_map, cols4 = gather_unique(idx4)  # uniq (K,), cols4 (Sb,4)

            ws_years: List[np.ndarray] = []

            for y in sorted(myrs_by_year.keys()):
                myr = myrs_by_year[y]
                if exact:
                    # Exact native height read
                    wsK = _read_var(myr, f"windspeed_{h_lo}m", uniq_idx)  # (T,K)
                    wdK = _read_var(myr, f"winddirection_{h_lo}m", uniq_idx)
                    uK, vK = uv_from_ws_wd(wsK, wdK)                       # (T,K)
                    T, K = uK.shape
                    u_g = uK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                    v_g = vK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                    w = w4[None, :, :]
                    u_s = (u_g * w).sum(axis=2)
                    v_s = (v_g * w).sum(axis=2)
                    ws_s = np.hypot(u_s, v_s).astype("float32")
                    ws_years.append(ws_s)
                else:
                    # Bracket: read lo & hi; if z below min or above max, this acts as extrapolation
                    wsK_lo = _read_var(myr, f"windspeed_{h_lo}m", uniq_idx)
                    wdK_lo = _read_var(myr, f"winddirection_{h_lo}m", uniq_idx)
                    wsK_hi = _read_var(myr, f"windspeed_{h_hi}m", uniq_idx)
                    wdK_hi = _read_var(myr, f"winddirection_{h_hi}m", uniq_idx)

                    uK_lo, vK_lo = uv_from_ws_wd(wsK_lo, wdK_lo)
                    uK_hi, vK_hi = uv_from_ws_wd(wsK_hi, wdK_hi)

                    T, K = uK_lo.shape
                    def gather(uK: np.ndarray, vK: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
                        u_g = uK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                        v_g = vK[:, cols4.reshape(-1)].reshape(T, len(batch), NBR_K)
                        w = w4[None, :, :]
                        u_s = (u_g * w).sum(axis=2)
                        v_s = (v_g * w).sum(axis=2)
                        return u_s, v_s

                    u_lo, v_lo = gather(uK_lo, vK_lo)
                    u_hi, v_hi = gather(uK_hi, vK_hi)

                    ws_lo = np.hypot(u_lo, v_lo, dtype="float64")
                    ws_hi = np.hypot(u_hi, v_hi, dtype="float64")

                    invalid = (~np.isfinite(ws_lo)) | (~np.isfinite(ws_hi)) | (ws_lo <= 0) | (ws_hi <= 0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        alpha = np.log(ws_lo/ws_hi) / np.log(float(h_lo)/float(h_hi))
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
                    "dataset":    "HRRR CONUS",
                    "height_m":   float(z),
                    "interp":     f"{INTERP_SP}; vert={INTERP_V if not exact else 'exact'}",
                    "agg":        "native",
                    "years":      "2015-2022",
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
