"""Shared utilities for grid extraction modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from wem.utils.columns import choose_col
from wem.utils.logging import log
from wem.utils.spatial import to_xy_lcc


@dataclass
class Tiles:
    groups: Dict[int, np.ndarray]   # tile_id -> indices in ERA5 df


@dataclass
class NeighborPack:
    uniq_cols: np.ndarray           # (K_unique,) sorted asc
    W: sparse.csr_matrix            # (K_unique, S_tile) weights


def load_era5_grid(path: Path, log_prefix: str = "[GRID]") -> pd.DataFrame:
    """Load ERA5 grid CSV/Parquet -> DataFrame with grid_id, lat, lon."""
    log(f"{log_prefix} Loading ERA5 grid: {path}")
    if path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        try:
            df = pd.read_csv(path, usecols=lambda c: c.lower() in {
                "grid_id", "lat", "lon", "latitude", "longitude"
            })
        except Exception:
            df = pd.read_csv(path)

    latc = choose_col(df, ["lat", "latitude"])
    lonc = choose_col(df, ["lon", "longitude"])
    if not latc or not lonc:
        raise ValueError("ERA5 grid must have lat/lon columns.")
    if "grid_id" not in df.columns:
        gid = pd.Series(np.arange(len(df), dtype=int)).astype("string").str.zfill(6)
        df["grid_id"] = gid

    out = df.rename(columns={latc: "lat", lonc: "lon"})[["grid_id", "lat", "lon"]].copy()
    out["grid_id"] = out["grid_id"].astype("string")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out


def make_tiles(df: pd.DataFrame, tile_km: float) -> Tiles:
    """Partition ERA5 grid points into spatial tiles."""
    x, y = to_xy_lcc(df["lon"].to_numpy(), df["lat"].to_numpy())
    x0, y0 = np.min(x), np.min(y)
    step = tile_km * 1000.0
    tx = np.floor((x - x0) / step).astype(np.int64)
    ty = np.floor((y - y0) / step).astype(np.int64)
    tile_id = (tx << 32) | (ty & 0xffffffff)
    groups: Dict[int, np.ndarray] = {}
    for t in np.unique(tile_id):
        groups[int(t)] = np.where(tile_id == t)[0]
    return Tiles(groups=groups)


def contiguous_runs(sorted_idxs: np.ndarray) -> List[Tuple[int, int]]:
    """Given sorted ints, return [(start, end_exclusive), ...] for contiguous runs."""
    if len(sorted_idxs) == 0:
        return []
    runs: List[Tuple[int, int]] = []
    start = int(sorted_idxs[0])
    prev = start
    for v in map(int, sorted_idxs[1:]):
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev + 1))
        start = v
        prev = v
    runs.append((start, prev + 1))
    return runs
