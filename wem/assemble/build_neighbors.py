#!/usr/bin/env python3
"""
Add leave-one-out exclusion lists for GoldStandard sites.

For each GoldStandard site, find all OTHER sites (ASOS + GS) within a fixed
radius (default 10 km) and store their station_ids in a new column
'neighbors_10km_site_ids'. Also store 'neighbors_10km_count'.

Input (defaults to your current table):
  combined_quantiles_long_with_topo.csv
Output:
  combined_quantiles_long_with_topo_loocv.csv

Columns expected in input:
  - station_id (stringy ids OK)
  - lat, lon (decimal degrees)
  - observation_type (values like 'ASOS' or 'GS'/'GoldStandard')
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from wem.utils.logging import log
from wem.utils.sites import normalize_obs_type

EARTH_RADIUS_KM = 6371.0088

def pairwise_haversine_km(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """
    Vectorized pairwise great-circle distances (km).
    lat_rad, lon_rad: shape (N,)
    returns D with shape (N, N)
    """
    # Broadcasting differences
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    # Haversine
    sin_dlat = np.sin(dlat * 0.5)
    sin_dlon = np.sin(dlon * 0.5)
    a = sin_dlat**2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * sin_dlon**2
    # numerical safety
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def build_neighbor_lists(
    sites: pd.DataFrame, radius_km: float
) -> pd.DataFrame:
    """
    sites: unique sites table with columns ['station_id','lat','lon','observation_type']
    Returns df with ['station_id','neighbors_10km_site_ids','neighbors_10km_count'] for GS sites only.
    """
    # Coerce dtypes and sanity check
    ss = sites.copy()
    ss["station_id"] = ss["station_id"].astype(str)
    ss["lat"] = pd.to_numeric(ss["lat"], errors="coerce")
    ss["lon"] = pd.to_numeric(ss["lon"], errors="coerce")
    ss["observation_type"] = ss["observation_type"].astype(str).map(normalize_obs_type)

    ss = ss.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    N = len(ss)
    if N == 0:
        raise ValueError("No valid sites with lat/lon.")

    log(f"[INFO] Unique sites: {N} (ASOS: {(ss['observation_type']=='ASOS').sum()}, GS: {(ss['observation_type']=='GS').sum()})")

    # Radians
    lat_rad = np.deg2rad(ss["lat"].to_numpy(dtype="float64"))
    lon_rad = np.deg2rad(ss["lon"].to_numpy(dtype="float64"))

    # Pairwise distances once
    D = pairwise_haversine_km(lat_rad, lon_rad)

    # For GS sites: find neighbors within radius, EXCLUDING itself
    is_gs = (ss["observation_type"] == "GS").to_numpy()
    idx_gs = np.where(is_gs)[0]

    neighbor_strings = {}
    neighbor_counts = {}

    r = float(radius_km)
    for i in idx_gs:
        # within radius, exclude self
        mask = (D[i] <= r) & (np.arange(N) != i)
        nbr_ids = ss.loc[mask, "station_id"].astype(str).tolist()
        # sort for determinism
        nbr_ids_sorted = sorted(nbr_ids)
        neighbor_strings[ss.at[i, "station_id"]] = ",".join(nbr_ids_sorted)
        neighbor_counts[ss.at[i, "station_id"]] = len(nbr_ids_sorted)

    # Build mapping frame (only GS have entries)
    out = pd.DataFrame({
        "station_id": list(neighbor_strings.keys()),
        "neighbors_10km_site_ids": list(neighbor_strings.values()),
    })
    out["neighbors_10km_count"] = out["neighbors_10km_site_ids"].map(lambda s: 0 if s == "" else s.count(",") + 1)
    return out

def main():
    ap = argparse.ArgumentParser(description="Add 10-km neighbor lists for GoldStandard sites.")
    ap.add_argument("--infile", type=Path, default=Path("combined_quantiles_long_with_topo.csv"),
                    help="Input long table (CSV or Parquet) with station_id/lat/lon/observation_type.")
    ap.add_argument("--outfile", type=Path, default=Path("combined_quantiles_long_with_topo_loocv_10km.csv"),
                    help="Output table with neighbor columns added.")
    ap.add_argument("--radius-km", type=float, default=10.0,
                    help="Radius (km) to define the LOOCV exclusion neighborhood.")
    args = ap.parse_args()

    # Load
    path = args.infile
    if not path.exists():
        raise FileNotFoundError(path)
    log(f"[INFO] Loading input table: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        # avoid dtype issues on station_id
        df = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)

    # Validate required columns
    required = {"station_id", "lat", "lon", "observation_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    # Unique sites by station_id (stable first occurrence)
    # If a station_id appears multiple times with slightly different lat/lon,
    # take the median lat/lon to be robust.
    log("[INFO] Building unique site list …")
    uniq = (
        df.groupby("station_id", as_index=False)
          .agg({
              "name": "first" if "name" in df.columns else "first",
              "lat": "median",
              "lon": "median",
              "observation_type": "first",
          })
    )

    # Build neighbor mapping for GS
    nbr = build_neighbor_lists(uniq[["station_id", "lat", "lon", "observation_type"]], args.radius_km)

    # Merge back onto all rows by station_id
    log("[INFO] Merging neighbor lists back to long table …")
    out = df.merge(nbr, on="station_id", how="left")

    # For non-GS sites, set empty strings / zeros (keep it explicit)
    is_gs_mask = out["observation_type"].astype(str).map(normalize_obs_type).eq("GS")
    out.loc[~is_gs_mask, "neighbors_10km_site_ids"] = out.loc[~is_gs_mask, "neighbors_10km_site_ids"].fillna("")
    out.loc[~is_gs_mask, "neighbors_10km_count"] = out.loc[~is_gs_mask, "neighbors_10km_count"].fillna(0).astype(int)

    # Save
    log(f"[INFO] Writing output → {args.outfile}")
    if args.outfile.suffix.lower() == ".parquet":
        out.to_parquet(args.outfile, index=False)
    else:
        out.to_csv(args.outfile, index=False)

    # Quick summary
    gs_sites = uniq["observation_type"].astype(str).map(normalize_obs_type).eq("GS").sum()
    with_nbr = nbr["neighbors_10km_count"].gt(0).sum()
    log(f"[INFO] GS sites: {gs_sites} | GS with ≥1 neighbor within {args.radius_km:.1f} km: {with_nbr}")
    log("[INFO] Done.")

if __name__ == "__main__":
    main()
