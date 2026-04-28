"""Nearest-neighbor prediction lookup for sites.

Input A (grid means): site_mean_winds.(csv|parquet) with columns:
  lat, lon, height_m, mean_ms
Input B (sites): CSV with columns:
  lat, lon, height_m
Output: sites_with_pred.csv (original columns + pred)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import BallTree
except ImportError as e:
    raise SystemExit(
        "This script requires scikit-learn. Install with `pip install scikit-learn`."
    ) from e

from wem.utils.logging import log


def to_rad(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon arrays from degrees to radians as a (N, 2) array."""
    return np.column_stack([np.deg2rad(lat_deg), np.deg2rad(lon_deg)])


def nearest_available_heights(
    available: np.ndarray, query_h: np.ndarray
) -> np.ndarray:
    """Map each query height to the nearest height in *available*.

    Parameters
    ----------
    available : np.ndarray
        1-D array of available heights, shape ``(H,)``.
    query_h : np.ndarray
        1-D array of query heights, shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Nearest available height for each query, shape ``(N,)``.
    """
    avail = available.reshape(-1, 1)  # (H, 1)
    diffs = np.abs(avail - query_h.reshape(1, -1))  # (H, N)
    idx = np.argmin(diffs, axis=0)  # (N,)
    return available[idx]


def main():
    ap = argparse.ArgumentParser(
        description="Nearest-neighbor prediction for sites "
        "(horizontal NN + nearest height)."
    )
    ap.add_argument(
        "--grid",
        type=Path,
        required=True,
        help="site_mean_winds.(csv|parquet) from wem-grid-means",
    )
    ap.add_argument(
        "--sites",
        type=Path,
        required=True,
        help="CSV of sites with columns lat,lon,height_m",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("sites_with_pred.csv"),
        help="Output CSV with added 'pred' column (m/s)",
    )
    args = ap.parse_args()

    # --- Load grid means ---
    log(f"Loading grid means: {args.grid}")
    if args.grid.suffix.lower() in (".parquet", ".pq"):
        g = pd.read_parquet(args.grid)
    else:
        g = pd.read_csv(args.grid, low_memory=False)

    need_g = ["lat", "lon", "height_m", "mean_ms"]
    miss_g = [c for c in need_g if c not in g.columns]
    if miss_g:
        raise SystemExit(f"Grid file missing required columns: {miss_g}")

    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g["height_m"] = pd.to_numeric(g["height_m"], errors="coerce")
    g["mean_ms"] = pd.to_numeric(g["mean_ms"], errors="coerce")
    g = g.dropna(subset=["lat", "lon", "height_m", "mean_ms"]).reset_index(
        drop=True
    )

    avail_heights = np.sort(g["height_m"].unique().astype(float))
    log(
        f"Available heights in grid: "
        f"{list(map(int, np.round(avail_heights)))}"
    )

    # --- Load sites ---
    log(f"Loading sites: {args.sites}")
    s = pd.read_csv(args.sites)
    need_s = ["lat", "lon", "height_m"]
    miss_s = [c for c in need_s if c not in s.columns]
    if miss_s:
        raise SystemExit(f"Sites file missing required columns: {miss_s}")

    s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
    s["lon"] = pd.to_numeric(s["lon"], errors="coerce")
    s["height_m"] = pd.to_numeric(s["height_m"], errors="coerce")
    s = s.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)

    # --- Map each site to nearest available height ---
    qh = s["height_m"].to_numpy(dtype=float)
    nearest_h = nearest_available_heights(avail_heights, qh)
    s["_nearest_h"] = nearest_h

    # --- Build BallTrees per height we actually need ---
    preds = np.full(len(s), np.nan, dtype=float)
    for h in np.unique(nearest_h):
        hmask = s["_nearest_h"] == h
        gh = g.loc[np.isclose(g["height_m"], h)].copy()
        if gh.empty:
            continue
        G_rad = to_rad(gh["lat"].to_numpy(), gh["lon"].to_numpy())
        tree = BallTree(G_rad, metric="haversine")
        Q_rad = to_rad(
            s.loc[hmask, "lat"].to_numpy(), s.loc[hmask, "lon"].to_numpy()
        )
        dist, idx = tree.query(Q_rad, k=1)
        idx = idx.ravel()
        preds[hmask] = gh["mean_ms"].to_numpy()[idx]

    out = s.drop(columns=["_nearest_h"]).copy()
    out["pred"] = preds

    log(f"Writing → {args.out}")
    out.to_csv(args.out, index=False)
    log(f"Done. Rows: {len(out):,}")


if __name__ == "__main__":
    main()
