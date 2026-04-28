"""Site-list loading, resume-tracking, and observation-type normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wem.utils.columns import choose_col
from wem.utils.logging import log


def load_sites(path: Path) -> pd.DataFrame:
    """Load a sites CSV and normalize columns to (station_id, name, lat, lon, elev_m).

    Flexibly detects column names via case-insensitive matching.  Drops rows
    with missing lat/lon and deduplicates by station_id.
    """
    log(f"[INFO] Loading sites CSV: {path}")
    df = pd.read_csv(path, dtype=str)

    idc = choose_col(df, ["station_id", "site_id", "STATION", "id"])
    latc = choose_col(df, ["lat", "LAT", "Latitude"])
    lonc = choose_col(df, ["lon", "LON", "Longitude"])
    namec = choose_col(df, ["name", "NAME", "station_name", "site_name"])
    elevc = choose_col(df, ["elev_m", "elevation_m", "elevation_meters"])

    if not (idc and latc and lonc):
        raise ValueError("Sites CSV must include station_id, lat, lon.")

    out = df.rename(columns={
        idc: "station_id",
        latc: "lat",
        lonc: "lon",
        (namec or idc): "name",
        (elevc or idc): "elev_m",
    })[["station_id", "name", "lat", "lon", "elev_m"]].drop_duplicates("station_id")

    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["elev_m"] = pd.to_numeric(out["elev_m"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    log(f"[INFO] Sites loaded: {len(out)}")
    return out


def already_done(out_csv: Path) -> set[str]:
    """Return the set of station_ids already present in *out_csv*.

    Used for resume-safe scripts that append rows incrementally.
    Returns an empty set if the file does not exist or cannot be read.
    """
    if out_csv.exists():
        try:
            done = set(pd.read_csv(out_csv, usecols=["station_id"], dtype=str)["station_id"])
            log(f"[INFO] Found existing output: {out_csv} (already has {len(done)} rows)")
            return done
        except Exception:
            return set()
    return set()


def load_gs_sites(path: Path) -> pd.DataFrame:
    """Load a Gold Standard sites CSV with height_m column.

    Like :func:`load_sites` but keeps the ``height_m`` column and
    deduplicates by ``(station_id, height_m)`` instead of ``station_id``
    alone.

    Returns
    -------
    pd.DataFrame
        Columns: ``station_id``, ``name``, ``lat``, ``lon``, ``elev_m``,
        ``height_m``.
    """
    log(f"[INFO] Loading GS sites CSV: {path}")
    df = pd.read_csv(path, dtype=str)

    idc = choose_col(df, ["station_id", "site_id", "STATION", "id"])
    latc = choose_col(df, ["lat", "LAT", "Latitude"])
    lonc = choose_col(df, ["lon", "LON", "Longitude"])
    namec = choose_col(df, ["name", "NAME", "station_name", "site_name"])
    elevc = choose_col(df, ["elev_m", "ELEV(M)", "elevation", "elevation_m", "elevation_meters"])
    hc = choose_col(df, ["height_m", "height", "z"])

    if not (idc and latc and lonc and hc):
        raise ValueError("GS sites CSV must include station_id, lat, lon, height_m.")

    out = df.rename(columns={
        idc: "station_id",
        latc: "lat",
        lonc: "lon",
        (namec or idc): "name",
        (elevc or idc): "elev_m",
        hc: "height_m",
    })[["station_id", "name", "lat", "lon", "elev_m", "height_m"]].copy()

    out["station_id"] = out["station_id"].astype(str)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["elev_m"] = pd.to_numeric(out["elev_m"], errors="coerce")
    out["height_m"] = pd.to_numeric(out["height_m"], errors="coerce")

    out = out.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)
    out = out.drop_duplicates(subset=["station_id", "height_m"]).reset_index(drop=True)
    log(f"[INFO] GS sites loaded: {len(out)} rows (unique station+height pairs)")
    return out


def already_done_gs(out_csv: Path) -> set[tuple[str, float]]:
    """Return ``{(station_id, height_m), ...}`` pairs already in *out_csv*.

    Gold Standard variant of :func:`already_done` that tracks progress
    per (station, height) rather than per station alone.
    """
    if out_csv.exists():
        try:
            tmp = pd.read_csv(
                out_csv,
                usecols=["station_id", "height_m"],
                dtype={"station_id": str, "height_m": float},
            )
            done = {(str(s), float(h)) for s, h in zip(tmp["station_id"], tmp["height_m"])}
            log(f"[INFO] Found existing output: {out_csv} (already has {len(done)} rows)")
            return done
        except Exception:
            return set()
    return set()


def normalize_obs_type(s: str) -> str:
    """Normalize an observation_type string to canonical form ('ASOS' or 'GS').

    Recognizes common variants such as 'goldstandard', 'gold standard',
    'gold_stand', 'gold-std', etc.  Returns the original (stripped) value
    if no known alias matches.
    """
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    if t in {"gs", "gold", "goldstandard", "gold standard", "gold_stand", "gold-std"}:
        return "GS"
    if t in {"asos"}:
        return "ASOS"
    return s.strip()
