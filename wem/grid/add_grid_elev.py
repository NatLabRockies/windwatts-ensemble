#!/usr/bin/env python3
"""
add_grid_elev.py  --  EXACT sampling behavior as training script, with debug & retry.

What stays identical to the training script:
- ArcGIS ImageServer /identify on 3DEP elevation
- Web Mercator (EPSG:3857) geometry & outSR
- Mosaic rule: esriMosaicNorthwest
- Pixel sizes tried IN ORDER: 10 m, then 30 m, then 90 m (stop on first success)
- Default: interpolateValues = false (match training script default)
- Attach one elevation per unique (lon,lat) and merge back to the full table

New (debug-friendly, safe to leave off):
- Retries 429/5xx with exponential backoff (honors Retry-After)
- Failure reason classification and end-of-run summary
- --debug prints sample failures; --fail-log writes a CSV of failures
- Optional Kansas-land box helper for sanity checks

Usage (same flags you used):
  python add_grid_elev.py \\
    --in merged_quantiles_all.csv \\
    --out merged_quantiles_all_with_elev.csv \\
    --workers 32 --timeout 15 --overwrite

Optional debug:
  python add_grid_elev.py \\
    --in merged_quantiles_all.csv \\
    --out merged_quantiles_all_with_elev.csv \\
    --workers 32 --timeout 15 --overwrite \\
    --debug --fail-log elev_failures.csv
"""

from __future__ import annotations
import argparse
import concurrent.futures as cf
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List
from collections import Counter
import threading
import random

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from wem.utils.logging import log
from wem.utils.spatial import to_webmercator

SERVICE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
IDENTIFY = SERVICE + "/identify"
MOSAIC_RULE = {"mosaicMethod": "esriMosaicNorthwest"}

# Optional: central Kansas all-land debug box (EPSG:3857)
X_MIN_KS = -11187608.824724
X_MAX_KS = -10964969.843137
Y_MIN_KS =  4579425.812870
Y_MAX_KS =  4793547.459105
def in_kansas_box_3857(x: float, y: float) -> bool:
    return (X_MIN_KS <= x <= X_MAX_KS) and (Y_MIN_KS <= y <= Y_MAX_KS)

# ---- Failure tracking (thread-safe)
_fail_lock = threading.Lock()
fail_counts = Counter()
fail_samples: Dict[str, List[dict]] = {}
MAX_SAMPLES_PER_REASON = 10

def record_failure(reason: str, rec: dict):
    with _fail_lock:
        fail_counts[reason] += 1
        lst = fail_samples.setdefault(reason, [])
        if len(lst) < MAX_SAMPLES_PER_REASON:
            lst.append(rec)

# ---- Identify helper with retry & classification
def identify_point_3857(
    session: requests.Session,
    x_merc: float,
    y_merc: float,
    pixel_size_m: float,
    timeout: float,
    interpolate: bool,
    max_retries: int = 4,
    backoff_base: float = 0.75,
    backoff_jitter: float = 0.25,
    debug: bool = False,
) -> Tuple[Optional[float], Optional[str], Optional[int], Optional[str]]:
    """
    Returns (value, reason, http_status, message)
      - value: float or None
      - reason: None if success, else failure category string
    Retries on 429/5xx with exponential backoff.
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

    attempt = 0
    while True:
        try:
            r = session.post(IDENTIFY, data=payload, timeout=timeout)
            status = r.status_code
            if status == 429 or 500 <= status < 600:
                # retryable
                reason = "http_error"
                msg = f"HTTP {status}"
                if attempt < max_retries:
                    wait = float(r.headers.get("Retry-After", 0)) or (backoff_base * (2 ** attempt) + random.random()*backoff_jitter)
                    if debug:
                        print(f"[debug] retry {attempt+1}/{max_retries} after {wait:.2f}s for {msg}")
                    time.sleep(min(wait, 10.0))
                    attempt += 1
                    continue
                else:
                    return None, reason, status, msg

            # non-retry path
            r.raise_for_status()
            try:
                js = r.json()
            except Exception as e:
                return None, "json_error", status, f"json decode: {e}"

            if "error" in js:
                # ArcGIS service-level error
                err = js.get("error", {})
                msg = f"{err.get('message','service error')}"
                return None, "service_error", status, msg

            v = js.get("value", None)
            if v is None:
                return None, "no_value", status, "value=None"

            try:
                vnum = float(v)
            except Exception:
                return None, "json_error", status, f"value-not-float:{v!r}"

            if not math.isfinite(vnum):
                return None, "non_finite", status, f"value={vnum}"

            return vnum, None, status, None

        except requests.Timeout:
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt) + random.random()*backoff_jitter
                if debug:
                    print(f"[debug] timeout, retry {attempt+1}/{max_retries} after {wait:.2f}s")
                time.sleep(min(wait, 10.0))
                attempt += 1
                continue
            return None, "timeout", None, "timeout"
        except Exception as e:
            # network error, etc.; not retrying beyond configured policy
            return None, "exception", None, repr(e)

def identify_with_fallbacks(
    x_merc: float,
    y_merc: float,
    session: requests.Session,
    timeout: float,
    pixel_sizes: Tuple[float, ...],
    interpolate: bool,
    debug: bool,
) -> Tuple[Optional[float], Optional[str], Optional[int], Optional[str], Tuple[float,float]]:
    """
    Try 10m -> 30m -> 90m. Return (value, reason, http_status, message, (x,y)).
    Stops at first success.
    """
    last_reason = last_status = last_message = None
    for px in pixel_sizes:
        val, reason, status, msg = identify_point_3857(
            session, x_merc, y_merc, px, timeout, interpolate, debug=debug
        )
        if val is not None:
            return val, None, status, None, (x_merc, y_merc)
        last_reason, last_status, last_message = reason, status, msg
        if debug:
            print(f"[debug] px={px}m -> None ({reason}, {status}, {msg})")
    return None, last_reason, last_status, last_message, (x_merc, y_merc)

def sample_elevation_points(
    pts_lonlat: Iterable[Tuple[float, float]],
    workers: int,
    timeout: float,
    pixel_sizes: Tuple[float, ...],
    interpolate: bool,
    debug: bool,
    log_kansas_checks: bool,
) -> Dict[Tuple[float, float], Optional[float]]:
    """Sample elevation at each lon/lat (WGS84) via /identify in EPSG:3857."""
    pts = list(pts_lonlat)
    out: Dict[Tuple[float, float], Optional[float]] = {}

    with requests.Session() as session:
        # small warmup
        try:
            session.get(SERVICE, timeout=5)
        except Exception:
            pass

        with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futs = {}
            for (lon, lat) in pts:
                xm, ym = to_webmercator(float(lon), float(lat))
                fut = pool.submit(
                    identify_with_fallbacks, xm, ym, session, timeout, pixel_sizes, interpolate, debug
                )
                futs[fut] = (lon, lat, xm, ym)

            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Sampling elevation", unit="pt"):
                lon, lat, xm, ym = futs[fut]
                try:
                    val, reason, status, msg, (x, y) = fut.result()
                    if val is not None:
                        out[(lon, lat)] = val
                    else:
                        out[(lon, lat)] = None
                        rec = {"lon": lon, "lat": lat, "x": x, "y": y, "reason": reason, "http_status": status, "message": msg}
                        if log_kansas_checks:
                            rec["in_kansas_box"] = bool(in_kansas_box_3857(x, y))
                        record_failure(reason or "unknown", rec)
                except Exception as e:
                    out[(lon, lat)] = None
                    rec = {"lon": lon, "lat": lat, "x": xm, "y": ym, "reason": "exception", "http_status": None, "message": repr(e)}
                    if log_kansas_checks:
                        rec["in_kansas_box"] = bool(in_kansas_box_3857(xm, ym))
                    record_failure("exception", rec)

    return out

def main():
    ap = argparse.ArgumentParser(description="Add USGS 3DEP elevation (meters) to a quantiles table -- training-parity + debug.")
    ap.add_argument("--in",  dest="infile",  type=Path, required=True)
    ap.add_argument("--out", dest="outfile", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16, help="Concurrent requests (threads)")
    ap.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout (s)")
    ap.add_argument("--pixel-sizes", type=str, default="10,30,90",
                    help="Comma-separated pixel sizes (meters) to try in order (default 10,30,90).")
    ap.add_argument("--interpolate", action="store_true", default=False,
                    help="Match training default: interpolation OFF unless you pass this flag.")
    ap.add_argument("--stream-chunksize", type=int, default=0,
                    help="If >0 and input is CSV, stream-merge in chunks of this many rows.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    # debug extras
    ap.add_argument("--debug", action="store_true", help="Verbose debug prints for failures and retries.")
    ap.add_argument("--fail-log", type=Path, default=None, help="Optional CSV path to write failed samples.")
    ap.add_argument("--kansas-check", action="store_true", help="Annotate failures with in_kansas_box boolean.")
    args = ap.parse_args()

    if args.outfile.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.outfile} (use --overwrite).")

    # Parse pixel sizes
    try:
        px = tuple(float(x) for x in args.pixel_sizes.split(",") if str(x).strip() != "")
    except Exception:
        px = (10.0, 30.0, 90.0)

    # Step 1: discover unique lon/lat (memory-light)
    log(f"Loading lon/lat from: {args.infile}")
    ext = args.infile.suffix.lower()
    if ext == ".parquet":
        lonlat = pd.read_parquet(args.infile, columns=["lon", "lat"])
    else:
        lonlat = pd.read_csv(args.infile, usecols=["lon", "lat"], dtype={"lon": "float64", "lat": "float64"})

    lonlat["lon"] = pd.to_numeric(lonlat["lon"], errors="coerce")
    lonlat["lat"] = pd.to_numeric(lonlat["lat"], errors="coerce")
    lonlat = lonlat.dropna(subset=["lon", "lat"])
    uniq = lonlat.drop_duplicates().reset_index(drop=True)
    del lonlat
    log(f"Unique coordinate pairs to sample: {len(uniq):,}")

    # Step 2: sample elevation (training-parity behavior; plus robust debug/retry)
    pts = list(map(tuple, uniq[["lon", "lat"]].to_numpy(dtype=float)))
    elev_map = sample_elevation_points(
        pts, workers=args.workers, timeout=args.timeout,
        pixel_sizes=px, interpolate=args.interpolate,
        debug=args.debug, log_kansas_checks=args.kansas_check
    )

    uniq["elevation_m"] = [elev_map.get((lon, lat), None) for lon, lat in pts]
    ne = int(uniq["elevation_m"].isna().sum())
    log(f"Elevation coverage: {len(uniq)-ne:,}/{len(uniq):,} unique points")

    # Optional: write failures CSV
    if args.fail_log is not None:
        with args.fail_log.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["lon","lat","x","y","reason","http_status","message","in_kansas_box"])
            w.writeheader()
            # flatten samples (not all failures -- just sample per reason)
            for reason, samples in fail_samples.items():
                for rec in samples:
                    if "in_kansas_box" not in rec:
                        rec["in_kansas_box"] = ""
                    w.writerow(rec)
        log(f"Wrote sample failures -> {args.fail_log}")

    # Step 3: merge back and save
    log(f"Merging elevation back -> {args.outfile}")
    if ext == ".parquet":
        df = pd.read_parquet(args.infile)
        out = df.merge(uniq, on=["lon", "lat"], how="left")
        out.to_parquet(args.outfile, index=False)
    else:
        if args.stream_chunksize and args.stream_chunksize > 0:
            header_written = False
            uniq_indexed = uniq.set_index(["lon", "lat"])
            for chunk in pd.read_csv(args.infile, chunksize=args.stream_chunksize, low_memory=False):
                chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
                chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
                chunk = chunk.join(uniq_indexed, on=["lon", "lat"])
                mode = "w" if not header_written else "a"
                chunk.to_csv(args.outfile, mode=mode, header=not header_written, index=False)
                header_written = True
        else:
            df = pd.read_csv(args.infile, low_memory=False)
            out = df.merge(uniq, on=["lon", "lat"], how="left")
            out.to_csv(args.outfile, index=False)

    # Report summary of failures by category
    if fail_counts:
        log("---- Failure summary (unique-point sampling) ----")
        for k, v in fail_counts.most_common():
            log(f"{k:>12}: {v:,}")
        if args.debug:
            for reason, samples in fail_samples.items():
                log(f"[debug] samples for {reason}:")
                for rec in samples:
                    log(f"  {rec}")
    else:
        log("No sampling failures recorded.")

    # Final NaN report
    if args.outfile.suffix.lower() == ".parquet":
        final = pd.read_parquet(args.outfile, columns=["elevation_m"])
    else:
        final = pd.read_csv(args.outfile, usecols=["elevation_m"])
    nan_elev = int(final["elevation_m"].isna().sum())
    log(f"Done. Rows: {len(final):,} | elevation_m NaNs: {nan_elev:,}")
    log("Note: 3DEP elevation is DEM (bare earth) in meters).")

if __name__ == "__main__":
    main()
