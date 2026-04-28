#!/usr/bin/env python3
# merge_3dep_topography_imageserver.py  (Web Mercator + pixelSize + server funcs)
#
# Samples USGS 3DEP elevation (DEM, meters), slope (degrees), and aspect (degrees)
# using the ImageServer /identify operation. Points are sent in EPSG:3857 with an explicit
# 10 m pixelSize and a simple mosaicRule to ensure consistent sampling.
#
# Usage:
#   python add_topography.py \
#       --in combined_quantiles_long.csv \
#       --out combined_quantiles_long_with_topo.csv \
#       --workers 16 --timeout 15
#
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from wem.utils.logging import log
from wem.utils.spatial import to_webmercator

SERVICE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
IDENTIFY = SERVICE + "/identify"

# ----- ArcGIS rendering rules straight from the service -----
RR_SLOPE  = {"rasterFunction": "Slope Degrees",  "rasterFunctionArguments": {"zFactor": 1}}
RR_ASPECT = {"rasterFunction": "Aspect Degrees", "rasterFunctionArguments": {}}

# Consistent mosaic rule (deterministic pick of source pixel)
MOSAIC_RULE = {"mosaicMethod": "esriMosaicNorthwest"}

def identify_point_3857(
    x_merc: float,
    y_merc: float,
    render_rule: Optional[dict] = None,
    timeout: float = 20.0,
    session: Optional[requests.Session] = None,
    pixel_size_m: float = 10.0,
    interpolate: bool = False,
) -> Optional[float]:
    """
    Call ImageServer /identify for a single Web Mercator point; returns numeric value or None.
    - geometry/outSR are 3857 (meters)
    - pixelSize is explicitly set in meters
    - optional renderingRule selects Slope/Aspect calculation server-side
    """
    geom = {"x": float(x_merc), "y": float(y_merc), "spatialReference": {"wkid": 3857}}
    payload = {
        "f": "json",
        "geometry": json.dumps(geom),
        "geometryType": "esriGeometryPoint",
        "outSR": json.dumps({"wkid": 3857}),
        "mosaicRule": json.dumps(MOSAIC_RULE),
        "pixelSize": json.dumps({"x": pixel_size_m, "y": pixel_size_m, "spatialReference": {"wkid": 3857}}),
        "returnGeometry": "false",
        "interpolateValues": "true" if interpolate else "false",
    }
    if render_rule is not None:
        payload["renderingRule"] = json.dumps(render_rule)

    s = session or requests.Session()
    try:
        r = s.post(IDENTIFY, data=payload, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            return None
        v = js.get("value", None)
        if v is None:
            return None
        vnum = float(v)
        return vnum if math.isfinite(vnum) else None
    except Exception:
        return None

def identify_with_fallbacks(
    x_merc: float,
    y_merc: float,
    render_rule: Optional[dict],
    session: requests.Session,
    timeout: float,
) -> Optional[float]:
    """Try a couple of pixel sizes in case we straddle a nodata boundary."""
    for px in (10.0, 30.0, 90.0):
        val = identify_point_3857(x_merc, y_merc, render_rule, timeout, session, pixel_size_m=px)
        if val is not None:
            return val
    return None

def sample_points(
    pts_lonlat: Iterable[Tuple[float, float]],
    render_rule: Optional[dict],
    workers: int,
    timeout: float,
) -> Dict[Tuple[float, float], Optional[float]]:
    """Sample each lon/lat with /identify (projected to 3857)."""
    pts = list(pts_lonlat)
    out: Dict[Tuple[float, float], Optional[float]] = {}

    with requests.Session() as session:
        try:
            session.get(SERVICE, timeout=5)
        except Exception:
            pass

        with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futs = {}
            for (lon, lat) in pts:
                xm, ym = to_webmercator(float(lon), float(lat))
                fut = pool.submit(identify_with_fallbacks, xm, ym, render_rule, session, timeout)
                futs[fut] = (lon, lat)

            label = "elev" if render_rule is None else ("slope" if render_rule is RR_SLOPE else "aspect")
            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc=f"Sampling {label}", unit="pt"):
                lon, lat = futs[fut]
                val = None
                try:
                    val = fut.result()
                except Exception:
                    val = None
                out[(lon, lat)] = val

    return out

def main():
    ap = argparse.ArgumentParser(description="Merge 3DEP elevation, slope, aspect via ImageServer /identify (EPSG:3857).")
    ap.add_argument("--in",  dest="infile",  type=Path, default=Path("combined_quantiles_long.csv"))
    ap.add_argument("--out", dest="outfile", type=Path, default=Path("combined_quantiles_long_with_topo.csv"))
    ap.add_argument("--workers", type=int, default=16, help="Concurrent requests")
    ap.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout (s)")
    args = ap.parse_args()

    # Load
    log(f"[INFO] Loading input table: {args.infile}")
    if args.infile.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.infile)
    else:
        df = pd.read_csv(args.infile, low_memory=False)

    # Clean lon/lat
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("Input must have 'lon' and 'lat' (degrees, WGS84).")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df_pts = df.dropna(subset=["lon", "lat"]).copy()
    log(f"[INFO] Rows with lon/lat: {len(df_pts):,}")

    uniq = df_pts[["lon", "lat"]].drop_duplicates().reset_index(drop=True)
    pts = list(map(tuple, uniq.to_numpy(dtype=float)))
    log(f"[INFO] Unique lon/lat points to sample: {len(pts):,}")

    # Sample three layers
    elev_map  = sample_points(pts, None,      workers=args.workers, timeout=args.timeout)
    slope_map = sample_points(pts, RR_SLOPE,  workers=args.workers, timeout=args.timeout)
    aspect_map= sample_points(pts, RR_ASPECT, workers=args.workers, timeout=args.timeout)

    # Attach results to unique table, merge back
    uniq["elevation_m"] = [elev_map.get((x, y))   for x, y in pts]
    uniq["slope_deg"]   = [slope_map.get((x, y))  for x, y in pts]
    uniq["aspect_deg"]  = [aspect_map.get((x, y)) for x, y in pts]

    out = df.merge(uniq, on=["lon", "lat"], how="left")

    # Save & report
    n = len(out)
    ne = int(out["elevation_m"].isna().sum())
    ns = int(out["slope_deg"].isna().sum())
    na = int(out["aspect_deg"].isna().sum())
    log(f"[INFO] Writing output → {args.outfile}")
    if args.outfile.suffix.lower() == ".parquet":
        out.to_parquet(args.outfile, index=False)
    else:
        out.to_csv(args.outfile, index=False)
    log(f"[INFO] Done. Rows: {n:,} | NaNs — elev:{ne:,} slope:{ns:,} aspect:{na:,}")
    log("[NOTE] Elevation is DEM height (bare earth, meters). It can differ from station metadata.")

if __name__ == "__main__":
    main()
