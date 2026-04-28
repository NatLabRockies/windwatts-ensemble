#!/usr/bin/env python3
"""
HRRR CONUS (2015-2022, 10 m) -> quantiles per site (q000..q100).

- Uses ERA5 site CSV for station list: expects columns station_id, lat, lon (name/elev optional)
- For each site: get 4 nearest grid points via myr.tree.query, do IDW on u,v, then WS
- Concatenate 2015-2022 native cadence and compute q000..q100
- Append rows to hrrr_quantiles_2015_2022.csv (resume-safe)

Usage (defaults work on your paths):
  python -m wem.extract.asos_hrrr \\
    --sites    era5_quantiles_2007_2024.csv \\
    --data-dir /datasets/WIND/HRRR \\
    --out      hrrr_quantiles_2015_2022.csv \\
    --batch    100 \\
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
from wem.utils.wind import uv_from_ws_wd, gather_unique, read_var
from wem.utils.spatial import idw_weights_from_dd

# ───────────────────── helpers ───────────────────────────────
def dt_index(myr: MultiYearWindX) -> pd.DatetimeIndex:
    dt = myr.time_index
    return dt if isinstance(dt, pd.DatetimeIndex) else pd.DatetimeIndex(dt)

def compute_quantiles(ws_all: np.ndarray) -> np.ndarray:
    # ws_all: (T_total, S) -> (101, S)
    qs = None
    try:
        qs = np.nanpercentile(ws_all, q=np.arange(101), axis=0, method="linear")
    except TypeError:
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
    ap = argparse.ArgumentParser(description="HRRR CONUS (2015-2022, 10m) -> quantiles per site.")
    ap.add_argument("--sites",    type=Path, default=Path("era5_quantiles_2007_2024.csv"), help="Site list CSV (ERA5)")
    ap.add_argument("--data-dir", type=Path, default=Path("/datasets/WIND/HRRR"),          help="Directory with hrrr_nat_f02_conus_YYYY.h5")
    ap.add_argument("--out",      type=Path, default=Path("hrrr_quantiles_2015_2022.csv"), help="Output CSV")
    ap.add_argument("--batch",    type=int,  default=100, help="Sites per batch")
    ap.add_argument("--log-every", type=int, default=200, help="Log 'appended X/Y' every N sites")
    args = ap.parse_args()

    DATASET = "HRRR CONUS"
    HEIGHT  = 10
    VAR_WS  = f"windspeed_{HEIGHT}m"
    VAR_WD  = f"winddirection_{HEIGHT}m"
    YEARS   = list(range(2015, 2023))          # inclusive 2015..2022
    YEARS_L = "2015-2022"
    AGG_L   = "native"

    # Sites & resume
    sites = load_sites(args.sites)
    done = already_done(args.out)
    todo = sites[~sites["station_id"].isin(done)].reset_index(drop=True)
    log(f"[INFO] {len(sites)} sites total; {len(already)} already in {args.out.name}; {len(todo)} to process.")
    if todo.empty:
        log("[INFO] Nothing to do.")
        return

    # Resolve & open year files
    year_paths: List[Path] = []
    for y in YEARS:
        p = args.data_dir / f"hrrr_nat_f02_conus_{y}.h5"
        if p.exists():
            year_paths.append(p)
        else:
            log(f"[WARN] Missing year file: {p} (skipping)")
    if not year_paths:
        raise FileNotFoundError(f"No HRRR files found in {args.data_dir} for years {YEARS_L}")

    log(f"[INFO] Opening {len(year_paths)} HRRR files ({YEARS_L})...")
    years: List[int] = []
    myrs : List[MultiYearWindX] = []
    for p in year_paths:
        y = int("".join([c for c in p.name if c.isdigit()])[:4])
        years.append(y)
        myrs.append(MultiYearWindX(str(p), hsds=False))
        log(f"[INFO]  Opened {p}")

    # Precompute neighbors for ALL to-do sites from the first available year's tree
    tree = myrs[0].tree
    log(f"[INFO] Precomputing nearest-4 neighbors for {len(todo)} sites ...")
    coords = todo[["lat", "lon"]].to_numpy(dtype="float64")
    dd_all = np.empty((len(todo), 4), dtype="float64")
    ii_all = np.empty((len(todo), 4), dtype="int64")
    for i, (lat, lon) in enumerate(coords):
        dd, ii = tree.query((lat, lon), 4)
        dd_all[i] = dd
        ii_all[i] = ii
    w_all = idw_weights_from_dd(dd_all)  # (S,4)

    # Process in batches
    S = len(todo)
    bs = max(1, int(args.batch))
    nb = int(np.ceil(S / bs))
    log(f"[INFO] Processing {S} sites in {nb} batches of {bs} (IDW-4-u,v; native cadence) ...")

    pbar = tqdm(total=S, unit="site")
    for bi in range(nb):
        s0, s1 = bi*bs, min(S, (bi+1)*bs)
        batch = todo.iloc[s0:s1].reset_index(drop=True)
        idx4  = ii_all[s0:s1]      # (Sb,4)
        w4    = w_all[s0:s1]       # (Sb,4)
        uniq_idx, pos_map, cols4 = gather_unique(idx4)  # uniq (K,), cols4 (Sb,4)
        ws_years: List[np.ndarray] = []

        for y, myr in zip(years, myrs):
            dt = dt_index(myr)
            t0, t1, step = 0, len(dt), 1

            # Read neighbors (T,K) for both variables
            wsK = read_var(myr, VAR_WS, t0, t1, step, uniq_idx)  # (T,K)
            wdK = read_var(myr, VAR_WD, t0, t1, step, uniq_idx)  # (T,K)

            uK, vK = uv_from_ws_wd(wsK, wdK)                    # (T,K)
            T, K = uK.shape
            # Gather neighbors into (T,Sb,4) using the K-positions in 'cols4'
            u_g = uK[:, cols4.reshape(-1)].reshape(T, len(batch), 4)
            v_g = vK[:, cols4.reshape(-1)].reshape(T, len(batch), 4)
            w = w4[None, :, :]                                   # (1,Sb,4)
            u_s = (u_g * w).sum(axis=2)                          # (T,Sb)
            v_s = (v_g * w).sum(axis=2)                          # (T,Sb)
            ws_s = np.sqrt(u_s*u_s + v_s*v_s, dtype="float32")   # (T,Sb)
            ws_years.append(ws_s)

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

        pbar.update(len(batch))
        if ((s1 % max(1, int(args.log_every))) == 0) or (s1 == S):
            log(f"[INFO] Progress: appended {s1}/{S}")

    pbar.close()
    log("[INFO] Complete.")

if __name__ == "__main__":
    main()
