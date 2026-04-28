#!/usr/bin/env python3
"""
WTK-LED Climate (North America, 2007-2020, 10 m) -> quantiles per site (q000..q100).

- Site list: ERA5 CSV with columns station_id, lat, lon (name/elev optional)
- For each site: 4-neighbor IDW in (u,v), then convert to speed
- Concatenate native time series for 2007..2020 and compute q000..q100
- Append rows to output CSV (resume-safe by station_id)

Usage:
  python -m wem.extract.asos_wtkled_climate \\
    --sites era5_quantiles_2007_2024.csv \\
    --out   wtk_led_climate_quantiles_2007_2020.csv \\
    --data-dir /datasets/WIND/ANL_4km_north_america \\
    --batch 100 \\
    --log-every 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from rex.resource_extraction import MultiYearWindX

from wem.utils.logging import log
from wem.utils.sites import already_done, load_sites
from wem.utils.wind import gather_unique, read_var
from wem.utils.spatial import idw_weights_from_dd

# ───────────────────── helpers ───────────────────────────────
def dt_index(myr: MultiYearWindX) -> pd.DatetimeIndex:
    dt = myr.time_index
    return dt if isinstance(dt, pd.DatetimeIndex) else pd.DatetimeIndex(dt)

def compute_quantiles(ws_all: np.ndarray) -> np.ndarray:
    # ws_all: (T_total, S) float32 -> (101, S)
    try:
        qs = np.nanpercentile(ws_all, q=np.arange(101), axis=0, method="linear")
    except TypeError:  # NumPy < 1.22 fallback
        qs = np.nanpercentile(ws_all, q=np.arange(101), axis=0, interpolation="linear")
    return qs.astype("float32")

def write_rows(out_csv: Path,
               sites_batch: pd.DataFrame,
               qs_batch: np.ndarray,
               dataset_label: str,
               height_m: int,
               agg_label: str,
               years_label: str) -> None:
    header_needed = not out_csv.exists()
    cols = [f"q{q:03d}" for q in range(101)]
    dfq = pd.DataFrame(qs_batch.T, columns=cols)   # (S,101)
    nowz = datetime.now(timezone.utc).isoformat().replace("+00:00", "+00:00Z")
    meta = pd.DataFrame({
        "station_id": sites_batch["station_id"].values,
        "name":       sites_batch["name"].values,
        "lat":        sites_batch["lat"].values,
        "lon":        sites_batch["lon"].values,
        "elev_m":     sites_batch["elev_m"].values,
        "dataset":    dataset_label,
        "height_m":   height_m,
        "interp":     "IDW-4-u,v",
        "agg":        agg_label,
        "years":      years_label,
        "processed_utc": nowz
    })
    out = pd.concat([meta, dfq], axis=1)
    out.to_csv(out_csv, mode="a", index=False, header=header_needed)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="WTK-LED Climate (2007-2020, 10m) -> quantiles per site.")
    ap.add_argument("--sites",    type=Path, default=Path("era5_quantiles_2007_2024.csv"), help="Site list CSV (ERA5)")
    ap.add_argument("--out",      type=Path, default=Path("wtk_led_climate_quantiles_2007_2020.csv"), help="Output CSV")
    ap.add_argument("--data-dir", type=Path, default=Path("/datasets/WIND/ANL_4km_north_america"),
                    help="Directory containing north_america_YYYY.h5 files")
    ap.add_argument("--batch",    type=int,  default=100, help="Sites per batch")
    ap.add_argument("--log-every", type=int, default=200, help="Log every N sites appended")
    args = ap.parse_args()

    DATASET = "WTK-LED Climate"
    HEIGHT  = 10
    VAR_WS  = f"windspeed_{HEIGHT}m"
    VAR_WD  = f"winddirection_{HEIGHT}m"
    YEARS   = list(range(2007, 2021))
    YEARS_L = "2007-2020"
    AGG_L   = "native"

    # Sites & resume
    sites = load_sites(args.sites)
    done = already_done(args.out)
    todo = sites[~sites["station_id"].isin(done)].reset_index(drop=True)
    log(f"[INFO] {len(sites)} sites total; {len(already)} already in {args.out.name}; {len(todo)} to process.")
    if todo.empty:
        log("[INFO] Nothing to do.")
        return

    # Resolve per-year files
    year_files: List[Path] = []
    for y in YEARS:
        p = args.data_dir / f"north_america_{y}.h5"
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        year_files.append(p)
    log(f"[INFO] Years: {YEARS[0]}-{YEARS[-1]} ({len(year_files)} files).")

    # Open the first year to build the KDTree (grid shared across years)
    lead = MultiYearWindX(str(year_files[0]), hsds=False)
    tree = lead.tree
    log("[INFO] Precomputing nearest-4 neighbors for all sites ...")
    coords = todo[["lat", "lon"]].to_numpy(dtype="float64")
    dd_all = np.empty((len(todo), 4), dtype="float64")
    ii_all = np.empty((len(todo), 4), dtype="int64")
    for i, (lat, lon) in enumerate(coords):
        dd, ii = tree.query((lat, lon), 4)
        dd_all[i] = dd
        ii_all[i] = ii
    w_all = idw_weights_from_dd(dd_all)  # (S,4)
    del lead  # free the handle

    # Process in batches
    S = len(todo)
    bs = max(1, int(args.batch))
    nb = int(np.ceil(S / bs))
    log(f"[INFO] Processing {S} sites in {nb} batches of {bs} (IDW-4-u,v; native cadence) ...")

    appended = 0
    pbar = tqdm(total=S, unit="site")
    for bi in range(nb):
        s0, s1 = bi*bs, min(S, (bi+1)*bs)
        batch = todo.iloc[s0:s1].reset_index(drop=True)
        idx4  = ii_all[s0:s1]      # (Sb,4)
        w4    = w_all[s0:s1]       # (Sb,4)
        uniq_idx, pos_map, cols4 = gather_unique(idx4)  # uniq (K,), cols4 (Sb,4)

        ws_years: List[np.ndarray] = []
        for p in year_files:
            myr = MultiYearWindX(str(p), hsds=False)
            dt = dt_index(myr)
            t0, t1, step = 0, len(dt), 1

            # Read (T,K) for both variables
            wsK = read_var(myr, VAR_WS, t0, t1, step, uniq_idx)  # (T,K)
            wdK = read_var(myr, VAR_WD, t0, t1, step, uniq_idx)  # (T,K)

            # Components in float64 (maintain precision through IDW + hypot)
            theta = np.deg2rad((270.0 - wdK) % 360.0, dtype="float64")
            ws64 = wsK.astype("float64")
            uK = -ws64 * np.sin(theta)                           # (T,K), float64
            vK = -ws64 * np.cos(theta)                           # (T,K), float64
            T, K = uK.shape
            # Gather neighbors per site into (T,Sb,4)
            u_g = uK[:, cols4.reshape(-1)].reshape(T, len(batch), 4)
            v_g = vK[:, cols4.reshape(-1)].reshape(T, len(batch), 4)
            w = w4[None, :, :]                                   # (1,Sb,4), float32
            # Weighted sum over 4 neighbors -> (T,Sb)
            u_s = (u_g * w).sum(axis=2)
            v_s = (v_g * w).sum(axis=2)
            ws_s = np.hypot(u_s, v_s).astype("float32")          # (T,Sb)
            ws_years.append(ws_s)

        # Concatenate all years and compute quantiles
        ws_all = np.concatenate(ws_years, axis=0)                # (T_total,Sb)
        q = compute_quantiles(ws_all)                            # (101,Sb)

        write_rows(
            out_csv=args.out,
            sites_batch=batch,
            qs_batch=q,
            dataset_label=DATASET,
            height_m=HEIGHT,
            agg_label=AGG_L,
            years_label=YEARS_L,
        )

        appended += len(batch)
        pbar.update(len(batch))
        if (appended % max(1, int(args.log_every))) == 0 or appended == S:
            log(f"[INFO] Progress: appended {appended}/{S}")

    pbar.close()
    log("[INFO] Complete.")

if __name__ == "__main__":
    main()
