"""USGS 3DEP elevation, slope, and aspect sampling via ArcGIS ImageServer.

Provides functions to query the 3DEP ImageServer ``/identify`` endpoint
for elevation (DEM, metres), slope (degrees), or aspect (degrees) at
individual lon/lat points projected to EPSG:3857 (Web Mercator).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import requests
from tqdm.auto import tqdm

from wem.utils.logging import log
from wem.utils.spatial import to_webmercator

# ---------------------------------------------------------------------------
# Service constants
# ---------------------------------------------------------------------------

SERVICE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
IDENTIFY = SERVICE + "/identify"
MOSAIC_RULE = {"mosaicMethod": "esriMosaicNorthwest"}

# ArcGIS rendering rules for slope and aspect
RR_SLOPE = {"rasterFunction": "Slope Degrees", "rasterFunctionArguments": {"zFactor": 1}}
RR_ASPECT = {"rasterFunction": "Aspect Degrees", "rasterFunctionArguments": {}}


# ---------------------------------------------------------------------------
# Core identify helper
# ---------------------------------------------------------------------------


def identify_point_3857(
    x_merc: float,
    y_merc: float,
    render_rule: Optional[dict] = None,
    timeout: float = 20.0,
    session: Optional[requests.Session] = None,
    pixel_size_m: float = 10.0,
    interpolate: bool = False,
) -> Optional[float]:
    """Call ImageServer ``/identify`` for a single Web Mercator point.

    Parameters
    ----------
    x_merc, y_merc : float
        Point coordinates in EPSG:3857 (metres).
    render_rule : dict or None
        Optional rendering rule to select a derived layer (e.g.
        ``RR_SLOPE`` for slope degrees, ``RR_ASPECT`` for aspect degrees).
        Pass ``None`` for raw DEM elevation.
    timeout : float
        Per-request HTTP timeout in seconds.
    session : requests.Session or None
        An existing session for connection reuse.  A temporary session is
        created if ``None``.
    pixel_size_m : float
        Explicit pixel size in metres sent to the server (default 10 m).
    interpolate : bool
        Whether to request bilinear interpolation from the server
        (``interpolateValues``).

    Returns
    -------
    float or None
        Numeric pixel value, or ``None`` on any failure.
    """
    geom = {"x": float(x_merc), "y": float(y_merc), "spatialReference": {"wkid": 3857}}
    payload = {
        "f": "json",
        "geometry": json.dumps(geom),
        "geometryType": "esriGeometryPoint",
        "outSR": json.dumps({"wkid": 3857}),
        "mosaicRule": json.dumps(MOSAIC_RULE),
        "pixelSize": json.dumps(
            {"x": pixel_size_m, "y": pixel_size_m, "spatialReference": {"wkid": 3857}}
        ),
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


# ---------------------------------------------------------------------------
# Fallback cascade across pixel sizes
# ---------------------------------------------------------------------------


def identify_with_fallbacks(
    x_merc: float,
    y_merc: float,
    render_rule: Optional[dict],
    session: requests.Session,
    timeout: float,
) -> Optional[float]:
    """Try several pixel sizes in case the point straddles a nodata boundary.

    Attempts 10 m, 30 m, and 90 m pixel sizes in order and returns the
    first successful value (or ``None`` if all fail).
    """
    for px in (10.0, 30.0, 90.0):
        val = identify_point_3857(x_merc, y_merc, render_rule, timeout, session, pixel_size_m=px)
        if val is not None:
            return val
    return None


# ---------------------------------------------------------------------------
# Threaded batch sampling
# ---------------------------------------------------------------------------


def sample_elevation_points(
    pts_lonlat: Iterable[Tuple[float, float]],
    workers: int,
    timeout: float,
    render_rule: Optional[dict] = None,
) -> Dict[Tuple[float, float], Optional[float]]:
    """Sample a 3DEP layer at each lon/lat point using threaded requests.

    Parameters
    ----------
    pts_lonlat : iterable of (lon, lat)
        Geographic coordinates (WGS-84 degrees).
    workers : int
        Number of concurrent request threads.
    timeout : float
        Per-request HTTP timeout in seconds.
    render_rule : dict or None
        Rendering rule to select a derived layer.  Pass ``None`` for
        raw DEM elevation, ``RR_SLOPE`` for slope, or ``RR_ASPECT``
        for aspect.

    Returns
    -------
    dict[(lon, lat), float | None]
        Mapping from input coordinates to sampled values.
    """
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
                fut = pool.submit(identify_with_fallbacks, xm, ym, render_rule, session, timeout)
                futs[fut] = (lon, lat)

            label = "elev" if render_rule is None else (
                "slope" if render_rule is RR_SLOPE else "aspect"
            )
            for fut in tqdm(
                cf.as_completed(futs), total=len(futs), desc=f"Sampling {label}", unit="pt"
            ):
                lon, lat = futs[fut]
                val = None
                try:
                    val = fut.result()
                except Exception:
                    val = None
                out[(lon, lat)] = val

    return out
