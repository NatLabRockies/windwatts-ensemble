"""Raster point-sampling utilities for the WEM pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def sample_raster_points(
    path: Optional[Path],
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """Sample a single-band GeoTIFF at (lon, lat) points.

    Parameters
    ----------
    path : Path or None
        Path to the GeoTIFF file.  If ``None``, returns an array of NaNs.
    lons : np.ndarray
        Longitudes of sample points.
    lats : np.ndarray
        Latitudes of sample points.

    Returns
    -------
    np.ndarray
        Float32 array of sampled values with NaNs for nodata/missing pixels.
    """
    import rasterio  # lazy import -- only needed when actually sampling

    if path is None:
        return np.full_like(lons, np.nan, dtype="float32")

    with rasterio.open(path) as ds:
        nodata = ds.nodata
        vals = list(ds.sample(zip(lons, lats)))
        arr = np.array([v[0] if len(v) else np.nan for v in vals], dtype="float32")

        if nodata is not None:
            arr = np.where(np.isclose(arr, nodata), np.nan, arr)

        # guard against non-physical values
        arr = np.where(arr <= -1e20, np.nan, arr)
        return arr
