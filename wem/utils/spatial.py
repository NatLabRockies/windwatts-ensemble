"""Spatial projection and distance utilities for the WEM pipeline.

Includes Lambert Conformal Conic projection, Web Mercator projection,
inverse-distance weighting, and pairwise haversine distance computation.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from pyproj import Proj

# ---------------------------------------------------------------------------
# Lambert Conformal Conic projection (used for tiling WTK/ERA5 grids)
# ---------------------------------------------------------------------------

_LCC = Proj(
    "+proj=lcc +lat_1=30 +lat_2=60 "
    "+lat_0=38.47240422490422 +lon_0=-96.0 "
    "+x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs"
)


def to_xy_lcc(
    lon_deg: np.ndarray, lat_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude (degrees) to Lambert Conformal Conic x/y (meters)."""
    x, y = _LCC(lon_deg, lat_deg)
    return np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")


# ---------------------------------------------------------------------------
# Web Mercator projection (EPSG:3857)
# ---------------------------------------------------------------------------

_R = 6378137.0  # WGS-84 sphere radius (meters)
_MAX_Y = 85.0511287798066  # latitude clamp for Web Mercator


def to_webmercator(lon: float, lat: float) -> Tuple[float, float]:
    """Convert longitude/latitude (degrees, WGS-84) to Web Mercator x/y (meters, EPSG:3857)."""
    lat = max(min(lat, _MAX_Y), -_MAX_Y)
    x = math.radians(lon) * _R
    y = math.log(math.tan((math.pi / 4.0) + math.radians(lat) / 2.0)) * _R
    return x, y


# ---------------------------------------------------------------------------
# Inverse-distance weighting
# ---------------------------------------------------------------------------


def idw_weights_from_dd(dd: np.ndarray) -> np.ndarray:
    """Compute normalised IDW weights from neighbor distances.

    Parameters
    ----------
    dd : np.ndarray
        Neighbor distances with shape ``(S, 4)`` as returned by
        ``tree.query``.

    Returns
    -------
    np.ndarray
        Normalized weights ``(S, 4)`` proportional to ``1/dd``.
        When a distance is exactly zero the corresponding neighbor
        receives weight 1 and all others receive 0.
    """
    w = np.empty_like(dd, dtype="float64")
    zeros = dd <= 0
    if zeros.any():
        w[:] = 0.0
        rows = np.where(zeros.any(axis=1))[0]
        for r in rows:
            k = int(np.argmax(zeros[r]))
            w[r, k] = 1.0
        nz = ~zeros.any(axis=1)
        inv = 1.0 / (dd[nz] + 1e-9)
        w[nz] = inv / inv.sum(axis=1, keepdims=True)
    else:
        inv = 1.0 / (dd + 1e-9)
        w = inv / inv.sum(axis=1, keepdims=True)
    return w.astype("float32")


# ---------------------------------------------------------------------------
# Pairwise haversine distances
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0088


def pairwise_haversine_km(
    lat_rad: np.ndarray, lon_rad: np.ndarray
) -> np.ndarray:
    """Vectorized pairwise great-circle distances in kilometres.

    Parameters
    ----------
    lat_rad : np.ndarray
        Latitudes in **radians**, shape ``(N,)``.
    lon_rad : np.ndarray
        Longitudes in **radians**, shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Distance matrix ``D`` with shape ``(N, N)`` in kilometres.
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
