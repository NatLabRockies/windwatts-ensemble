"""Wind-vector conversion and resource-data reading utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def uv_from_ws_wd(ws: np.ndarray, wd_met: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind speed and direction to u/v components.

    Meteorological (FROM) degrees are converted via::

        theta = (270 - wd) % 360
        u = -ws * sin(theta)
        v = -ws * cos(theta)

    Parameters
    ----------
    ws : np.ndarray
        Wind speed array (any shape).
    wd_met : np.ndarray
        Wind direction in meteorological degrees (same shape as *ws*).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(u, v)`` component arrays as float32.
    """
    theta = np.deg2rad((270.0 - wd_met) % 360.0, dtype="float64")
    u = (-ws.astype("float64") * np.sin(theta)).astype("float32")
    v = (-ws.astype("float64") * np.cos(theta)).astype("float32")
    return u, v


def gather_unique(idx4: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    """Deduplicate a (S, 4) neighbor-index array and build a position map.

    Parameters
    ----------
    idx4 : np.ndarray
        Neighbor indices with shape ``(S, 4)`` (integers).

    Returns
    -------
    uniq_idx : np.ndarray
        Sorted unique indices, shape ``(K,)``.
    pos_map : dict[int, int]
        Mapping from global index to position in *uniq_idx* (0..K-1).
    cols4 : np.ndarray
        Remapped positions with shape ``(S, 4)`` indexing into *uniq_idx*.
    """
    uniq_idx = np.unique(idx4.reshape(-1))
    pos_map = {int(g): i for i, g in enumerate(uniq_idx)}
    cols4 = np.vectorize(pos_map.get)(idx4).astype("int64")
    return uniq_idx, pos_map, cols4


def read_var(myr, var: str, t0: int, t1: int, step: int, idxs: np.ndarray) -> np.ndarray:
    """Read a variable slice from a MultiYearWindX resource object.

    Attempts a single vectorized read of shape ``(T, K)``.  Falls back to
    a per-index loop if the bulk read raises an exception.

    Parameters
    ----------
    myr : MultiYearWindX
        An open REX multi-year wind resource handle.
    var : str
        Variable name (e.g. ``'windspeed_10m'``).
    t0, t1, step : int
        Start, stop, and step for the time-axis slice.
    idxs : np.ndarray
        1-D array of spatial indices to read.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, K)`` as float32.
    """
    try:
        arr = myr[var, slice(t0, t1, step), idxs]  # (T,K) expected
        return np.asarray(arr, dtype="float32")
    except Exception:
        cols = []
        for i in idxs:
            a = myr[var, slice(t0, t1, step), int(i)]
            cols.append(np.asarray(a, dtype="float32"))
        return np.stack(cols, axis=1)
