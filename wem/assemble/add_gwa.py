#!/usr/bin/env python3
# add_gwa_to_wsavg.py
"""
Sample Global Wind Atlas GeoTIFFs at each site (lon,lat) and
add gwa_10, gwa_50, gwa_100, gwa_150 plus a power-law vertical interpolation
gwa_interp at the site-specific height_m.

Input:
  site_height_ws_avg.csv  (from step 1)
    station_id,name,lat,lon,elevation_m,height_m,ws_avg

Args (provide paths to your GWA rasters):
  --gwa10 /path/to/gwa_10m.tif
  --gwa50 /path/to/gwa_50m.tif
  --gwa100 /path/to/gwa_100m.tif
  --gwa150 /path/to/gwa_150m.tif

Output:
  site_height_ws_avg_with_gwa.csv
    ... + gwa_10,gwa_50,gwa_100,gwa_150,gwa_interp
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from wem.utils.raster import sample_raster_points
from wem.utils.power_law import power_law_interp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="infile",  type=Path, required=True, help="site_height_ws_avg.csv")
    ap.add_argument("--out", dest="outfile", type=Path, required=True, help="site_height_ws_avg_with_gwa.csv")
    ap.add_argument("--gwa10", type=Path, required=False, help="GWA 10m GeoTIFF")
    ap.add_argument("--gwa50", type=Path, required=False, help="GWA 50m GeoTIFF")
    ap.add_argument("--gwa100", type=Path, required=False, help="GWA 100m GeoTIFF")
    ap.add_argument("--gwa150", type=Path, required=False, help="GWA 150m GeoTIFF")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, low_memory=False)
    for c in ["lat", "lon", "height_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)

    lons = df["lon"].to_numpy(dtype="float64")
    lats = df["lat"].to_numpy(dtype="float64")

    # Sample each provided raster
    gwa_cols: Dict[str, Optional[Path]] = {
        "gwa_10": args.gwa10,
        "gwa_50": args.gwa50,
        "gwa_100": args.gwa100,
        "gwa_150": args.gwa150,
    }
    for col, tif in gwa_cols.items():
        df[col] = sample_raster_points(tif, lons, lats) if tif else np.nan

    # Interpolate to the site height via power law
    heights_available = [10.0, 50.0, 100.0, 150.0]
    colmap = {10.0: "gwa_10", 50.0: "gwa_50", 100.0: "gwa_100", 150.0: "gwa_150"}

    def do_interp(row) -> float:
        H = float(row["height_m"])
        hv = []
        for h in heights_available:
            val = row[colmap[h]]
            hv.append((h, val))
        return power_law_interp(H, hv)

    df["gwa_interp"] = df.apply(do_interp, axis=1)

    df.to_csv(args.outfile, index=False)
    print(f"Wrote {len(df):,} rows → {args.outfile}")

if __name__ == "__main__":
    main()
