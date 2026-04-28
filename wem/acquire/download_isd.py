#!/usr/bin/env python3
"""
Download US ISD wind data 2007-2024 (one CSV per station) and
append metadata *immediately* after each successful download,
so the job can be interrupted and resumed.

Outputs
-------
wind_data_by_station/<station_id>_wind_2007_2024.csv
us_wind_station_metadata_2007_2024.csv   (grows as the run progresses)

Migrated from: download_us_wind_isd.py
"""

from __future__ import annotations

import argparse
import csv, io, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Dict

import pandas as pd
import requests

from wem.utils.logging import log

# ── CONSTANTS (non-configurable) ─────────────────────────────
DATA_TYPES    = "WND"
PAGE_SIZE     = 1_000            # API max
RETRY_MAX     = 5
RETRY_BACKOFF = 5

API_BASE  = "https://www.ncei.noaa.gov/access/services/data/v1"
META_URL  = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
TOKEN     = os.getenv("NCEI_TOKEN")
HEADERS   = {"token": TOKEN} if TOKEN else {}

# ── THREAD-SAFE METADATA APPEND ──────────────────────────────
_META_LOCK = threading.Lock()

def append_metadata(row: pd.Series, meta_out: Path) -> None:
    """Append one row to meta_out, creating file with header if needed."""
    with _META_LOCK:
        header_needed = not meta_out.exists()
        with meta_out.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=row.index)
            if header_needed:
                writer.writeheader()
            writer.writerow(row.to_dict())

# ── LOAD & FILTER STATION LIST ───────────────────────────────
def load_us_metadata() -> pd.DataFrame:
    df = pd.read_csv(META_URL, dtype=str)
    df = df[df["CTRY"] == "US"].copy()
    df["USAF"] = df["USAF"].str.zfill(6)
    df["WBAN"] = df["WBAN"].str.zfill(5)
    df["station_id"] = df["USAF"] + df["WBAN"]
    df = df[(df["USAF"] != "999999") & (df["WBAN"] != "99999")].drop_duplicates("station_id")
    df["BEGIN_YR"] = df["BEGIN"].str[:4].astype(int)
    df["END_YR"]   = df["END"].str[:4].astype(int)
    return df.reset_index(drop=True)

def stations_to_process(out_dir: Path, meta_out: Path) -> pd.DataFrame:
    meta_all = load_us_metadata()

    done_ids: set[str] = set()
    if meta_out.exists():
        done_ids |= set(pd.read_csv(meta_out, dtype=str)["station_id"])
    # also honour any CSVs that already exist (e.g., earlier manual run)
    done_ids |= {p.stem.split("_")[0] for p in out_dir.glob("*_wind_2007_2024.csv")}

    remaining = meta_all[~meta_all["station_id"].isin(done_ids)].reset_index(drop=True)
    log(f"{len(remaining)} stations remaining ({len(done_ids)} already done).")
    return remaining

# ── NETWORK HELPERS ──────────────────────────────────────────
def fetch_page(session: requests.Session, params: Dict[str, str]) -> str:
    """GET one CSV page with retries/back-off; params already include offset+limit."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = session.get(API_BASE, params=params, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
        except Exception as exc:
            if attempt == RETRY_MAX:
                log(f"permanent failure {params['stations']} offset {params['offset']}: {exc}")
                return ""
            wait = RETRY_BACKOFF ** attempt
            log(f"{params['stations']} retry {attempt}/{RETRY_MAX} in {wait}s ({exc})")
            time.sleep(wait)
    return ""

# ── PER-STATION DOWNLOAD ────────────────────────────────────
def stream_station(row: pd.Series, out_dir: Path, meta_out: Path,
                   start_date: str, end_date: str) -> bool:
    """Download one station; append metadata on success; return True if done."""
    station_id = row.station_id
    if row.END_YR < 2007 or row.BEGIN_YR > 2024:
        return False

    dest = out_dir / f"{station_id}_wind_2007_2024.csv"
    if dest.exists():
        append_metadata(row, meta_out)      # make sure metadata captures prior run
        return True

    base_params = {
        "dataset": "global-hourly",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": DATA_TYPES,
        "format": "csv",
        "includeStationName": "1",
        "includeAttributes": "0",
        "sortfield": "date",
        "sortorder": "asc",
        "limit": PAGE_SIZE,
    }

    with requests.Session() as sess, dest.open("w", newline="") as fh:
        writer, total, offset, last_date = None, 0, 1, None

        while True:
            params = base_params | {"offset": offset}
            txt = fetch_page(sess, params)
            if not txt.strip():
                break

            rdr  = csv.DictReader(io.StringIO(txt))
            rows = list(rdr)
            if not rows:
                break

            # duplicate page guard
            if rows[0]["DATE"] == last_date:
                break
            last_date = rows[0]["DATE"]

            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=rdr.fieldnames)
                writer.writeheader()
            writer.writerows(rows)
            total += len(rows)

            if len(rows) < PAGE_SIZE:       # final page
                break
            offset += len(rows)

    if total:
        log(f"{station_id}: {total:,} rows")
        append_metadata(row, meta_out)
        return True

    dest.unlink(missing_ok=True)
    return False

# ── MAIN ─────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download US ISD wind data (one CSV per station), restart-safe.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("wind_data_by_station"),
                    help="Directory for per-station wind CSVs.")
    ap.add_argument("--meta-out", type=Path,
                    default=Path("us_wind_station_metadata_2007_2024.csv"),
                    help="Metadata CSV (grows as stations complete).")
    ap.add_argument("--start-date", type=str, default="2007-01-01",
                    help="Start date for data download (YYYY-MM-DD).")
    ap.add_argument("--end-date", type=str, default="2024-12-31",
                    help="End date for data download (YYYY-MM-DD).")
    ap.add_argument("--max-workers", type=int, default=12,
                    help="Number of parallel download threads.")
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

    todo = stations_to_process(args.out_dir, args.meta_out)
    if todo.empty:
        log("Nothing left to do. Exiting.")
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {
            pool.submit(stream_station, row, args.out_dir, args.meta_out,
                        args.start_date, args.end_date): row.station_id
            for _, row in todo.iterrows()
        }
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                fut.result()
            except Exception as e:
                log(f"{sid} crashed: {e}")

    log("Run finished (you can re-run to resume).")

if __name__ == "__main__":
    main()
