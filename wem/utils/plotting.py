"""Plotting utilities for wind-resource maps.

Provides custom colormaps (sequential and diverging), longitude wrapping,
US boundary masking, US geometry loading from Cartopy's Natural Earth
cache or a local shapefile, and shared cartopy/statistics helpers used
across analysis modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

# Cartopy (optional — some environments don't have it)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# ---------------------------------------------------------------------------
# Custom sequential colormap (0 -> 10+ m/s mean wind speed)
# ---------------------------------------------------------------------------


def make_custom_cmap() -> LinearSegmentedColormap:
    """Build the WEM sequential wind-speed colormap.

    Color ramp:
        very dark purple -> deep blue -> rich blue -> green ->
        yellow -> orange -> neon red, with purple for >= vmax
        and light grey for NaN.
    """
    colors = [
        "#1A0033",  # very dark deep purple
        "#26238A",  # purple-leaning deep blue
        "#1554A3",  # rich blue
        "#1FA444",  # vibrant green
        "#F6D92A",  # bright yellow
        "#F98C1F",  # vivid orange
        "#FF1744",  # neon red
    ]
    cmap = LinearSegmentedColormap.from_list("wind_0to10plus", colors, N=256)
    cmap.set_over("#8F3A63")   # >= vmax
    cmap.set_bad("#EEEEEE")    # NaN
    cmap.set_under("#FFFFFF")  # < vmin (shouldn't occur)
    return cmap


# ---------------------------------------------------------------------------
# Diverging diff colormap (ML - dataset)
# ---------------------------------------------------------------------------


def make_diff_cmap() -> LinearSegmentedColormap:
    """Build the WEM diverging difference colormap.

    Color ramp:
        Negative (red #C33732) -> white -> Positive (blue #5A8DC1).
        NaN values are rendered as light grey.
    """
    colors = ["#C33732", "#FFFFFF", "#5A8DC1"]
    cmap = LinearSegmentedColormap.from_list("diff_rwb_custom", colors, N=256)
    cmap.set_bad("#EEEEEE")
    return cmap


# ---------------------------------------------------------------------------
# Robust statistics for color-scale limits
# ---------------------------------------------------------------------------


def robust_limits(
    series_list: List[pd.Series], trim: float = 0.02
) -> Tuple[float, float]:
    """Compute trimmed (lo, hi) limits from a list of numeric Series.

    Concatenates all series, drops non-finite values, then returns the
    *trim* and *1-trim* percentiles.  Falls back to (0, 1) on empty input.
    """
    x = pd.concat(series_list, ignore_index=True)
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return (0.0, 1.0)
    lo = float(np.nanpercentile(x, 100 * trim))
    hi = float(np.nanpercentile(x, 100 * (1 - trim)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return lo, hi


def symmetric_bias_limit(
    bias_series: List[pd.Series], trim: float = 0.02
) -> float:
    """Compute a symmetric color-limit from pooled bias series.

    Returns *L* such that the bias colorbar spans [-L, +L].
    Falls back to 5.0 on empty input.
    """
    x = pd.concat(bias_series, ignore_index=True)
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return 5.0
    lo = float(np.nanpercentile(x, 100 * trim))
    hi = float(np.nanpercentile(x, 100 * (1 - trim)))
    L = max(abs(lo), abs(hi))
    if not np.isfinite(L) or L <= 0:
        L = float(np.nanmax(np.abs(x))) if not x.empty else 5.0
    return L


# ---------------------------------------------------------------------------
# Cartopy basemap setup
# ---------------------------------------------------------------------------


def setup_cartopy_axes(
    conus: bool,
    ne_res: str = "50m",
    figsize: Tuple[float, float] = (12, 7),
    dpi: int = 150,
):
    """Create a Matplotlib figure with a Cartopy PlateCarree basemap.

    The basemap includes white land, gray ocean/lakes, state boundaries,
    and coastline — consistent across all WEM map outputs.

    Parameters
    ----------
    conus : bool
        If True, crop the map extent to the contiguous US.
    ne_res : str
        Natural Earth feature resolution (``"110m"``, ``"50m"``, ``"10m"``).
    figsize : tuple
        Figure size in inches ``(width, height)``.
    dpi : int
        Figure resolution.

    Returns
    -------
    (fig, ax)
        Matplotlib Figure and GeoAxes.

    Raises
    ------
    RuntimeError
        If Cartopy is not installed.
    """
    if not HAS_CARTOPY:
        raise RuntimeError("Cartopy is not installed; cannot render maps.")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(
        cfeature.LAND.with_scale(ne_res),
        facecolor="#FFFFFF",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.OCEAN.with_scale(ne_res),
        facecolor="#cccccc",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale(ne_res),
        facecolor="#cccccc",
        edgecolor="none",
        linewidth=0.5,
        zorder=1,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale(ne_res),
        edgecolor="#666666",
        linewidth=0.6,
        zorder=2,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale(ne_res),
        edgecolor="#666666",
        linewidth=0.5,
        zorder=2,
    )
    states = NaturalEarthFeature(
        "cultural",
        "admin_1_states_provinces_lines",
        ne_res,
        edgecolor="#999999",
        facecolor="none",
    )
    ax.add_feature(states, linewidth=0.4, zorder=2)

    if conus:
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
    else:
        ax.set_extent([-170, -60, 15, 72], crs=ccrs.PlateCarree())

    return fig, ax


# ---------------------------------------------------------------------------
# Longitude wrapping
# ---------------------------------------------------------------------------


def wrap_lon180(lon: np.ndarray) -> np.ndarray:
    """Wrap longitude values into the [-180, 180) range.

    Non-finite values are preserved as NaN.
    """
    out = ((lon + 180.0) % 360.0) - 180.0
    out[~np.isfinite(lon)] = np.nan
    return out


# ---------------------------------------------------------------------------
# US geometry loading and point masking
# ---------------------------------------------------------------------------

_US_PREP = None  # prepared geometry (lazy, cached)


def _us_names() -> List[str]:
    """Canonical lowercase name variants for the United States."""
    return ["united states of america", "united states", "usa"]


def _build_us_prepared_from_cartopy_cache() -> Optional[object]:
    """Attempt to load a prepared US geometry from Cartopy's Natural Earth cache."""
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union
    from shapely.prepared import prep

    for res in ("50m", "110m"):
        try:
            shp_path = shpreader.natural_earth(
                resolution=res, category="cultural", name="admin_0_countries"
            )
            recs = list(shpreader.Reader(shp_path).records())
            polys = []
            for rec in recs:
                attrs = rec.attributes
                name = (
                    attrs.get("NAME_LONG")
                    or attrs.get("ADMIN")
                    or attrs.get("SOVEREIGNT")
                    or ""
                ).lower()
                if any(name == t for t in _us_names()):
                    polys.append(rec.geometry)
            if polys:
                return prep(unary_union(polys))
        except Exception:
            continue
    return None


def _build_us_prepared_from_path(path_like: Path) -> object:
    """Load a prepared US geometry from a local shapefile or GeoJSON."""
    from shapely.ops import unary_union
    from shapely.prepared import prep

    try:
        import geopandas as gpd
    except Exception as e:
        raise RuntimeError(
            "Reading a local shapefile/GeoJSON requires GeoPandas. "
            "Install with `pip install geopandas` (or use conda)."
        ) from e

    gdf = gpd.read_file(str(path_like))
    if gdf.empty or "geometry" not in gdf.columns:
        raise RuntimeError(f"Failed to read geometries from {path_like}")

    name_cols = [
        c
        for c in gdf.columns
        if str(c).lower() in {"name", "name_long", "admin", "sovereignt"}
    ]
    if name_cols:
        col = name_cols[0]
        sel = gdf[col].astype(str).str.lower().isin(_us_names())
        if sel.any():
            gdf = gdf.loc[sel]

    geom = unary_union(gdf.geometry.values)
    return prep(geom)


def _build_us_prepared(us_shapefile: Optional[Path]) -> object:
    """Load the US polygon from Cartopy cache or a local file."""
    geom = _build_us_prepared_from_cartopy_cache()
    if geom is not None:
        return geom
    if us_shapefile is not None:
        if not Path(us_shapefile).exists():
            raise FileNotFoundError(f"--us-shapefile not found: {us_shapefile}")
        return _build_us_prepared_from_path(Path(us_shapefile))
    raise RuntimeError(
        "Could not load U.S. polygons from Cartopy's Natural Earth cache. "
        "Supply a local file via --us-shapefile (e.g., ne_50m_admin_0_countries.shp "
        "or a U.S.-only GeoJSON). See Natural Earth downloads: "
        "https://www.naturalearthdata.com/downloads/50m-cultural-vectors/"
    )


def mask_points_to_us(
    lon: np.ndarray, lat: np.ndarray, us_shapefile: Optional[Path] = None
) -> np.ndarray:
    """Return a boolean mask indicating which points lie within the US boundary.

    Parameters
    ----------
    lon : np.ndarray
        Longitude values (degrees).
    lat : np.ndarray
        Latitude values (degrees).
    us_shapefile : Path or None
        Optional path to a local US polygon file, used as a fallback if
        Cartopy's Natural Earth cache is unavailable.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)`` where ``True`` means the point
        is inside the US boundary.
    """
    from shapely.geometry import Point

    global _US_PREP
    if _US_PREP is None:
        _US_PREP = _build_us_prepared(us_shapefile)
    lon_n = wrap_lon180(lon)
    n = lon_n.shape[0]
    mask = np.zeros(n, dtype=bool)
    CH = 20000
    for i in range(0, n, CH):
        j = min(i + CH, n)
        pts = (Point(lx, ly) for lx, ly in zip(lon_n[i:j], lat[i:j]))
        mask[i:j] = np.fromiter(
            (_US_PREP.contains(p) for p in pts), count=(j - i), dtype=bool
        )
    return mask
