"""Shared fixtures and auto-skip hooks for the WEM test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wem.constants import QCOLS


# ---------------------------------------------------------------------------
# Auto-skip tests when optional dependencies are missing
# ---------------------------------------------------------------------------

_MARKER_PKG = {
    "requires_cartopy": "cartopy",
    "requires_rasterio": "rasterio",
    "requires_rex": "rex",
    "requires_xgboost": "xgboost",
}

_DEPENDENCY_ERRORS = {}


def _dependency_error(pkg: str):
    if pkg not in _DEPENDENCY_ERRORS:
        try:
            __import__(pkg)
        except Exception as exc:
            _DEPENDENCY_ERRORS[pkg] = exc
        else:
            _DEPENDENCY_ERRORS[pkg] = None
    return _DEPENDENCY_ERRORS[pkg]


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker_name, pkg in _MARKER_PKG.items():
            if marker_name in item.keywords:
                exc = _dependency_error(pkg)
                if exc is not None:
                    item.add_marker(
                        pytest.mark.skip(reason=f"{pkg} unavailable: {exc}")
                    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def qcols():
    """List of 101 quantile column names (q000..q100)."""
    return list(QCOLS)


@pytest.fixture()
def rng():
    """Deterministic NumPy random generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_quantile_row(qcols):
    """Single-row DataFrame with q000..q100 = linspace(0, 10, 101)."""
    vals = np.linspace(0, 10, 101)
    return pd.DataFrame([dict(zip(qcols, vals))])


@pytest.fixture()
def synthetic_quantile_df(qcols):
    """Five-row DataFrame with different linear ramps in q000..q100."""
    rows = []
    for scale in [1.0, 2.0, 3.0, 5.0, 8.0]:
        rows.append(dict(zip(qcols, np.linspace(0, scale, 101))))
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_sites_df():
    """Ten synthetic CONUS sites with station_id, lat, lon, elev_m, name."""
    return pd.DataFrame({
        "station_id": [f"S{i:03d}" for i in range(10)],
        "name": [f"Site {i}" for i in range(10)],
        "lat": np.linspace(25, 48, 10),
        "lon": np.linspace(-120, -70, 10),
        "elev_m": np.linspace(10, 2000, 10),
    })


@pytest.fixture()
def synthetic_long_training_df(qcols):
    """Long-format training DataFrame: 3 stations x 2 heights x 101 qnums."""
    rows = []
    for sid, obs_type in [("GS001", "GS"), ("GS002", "GS"), ("ASOS001", "ASOS")]:
        for h in [60, 100]:
            for q in range(101):
                rows.append({
                    "station_id": sid,
                    "height_m": h,
                    "qnum": q,
                    "observation": np.random.default_rng(hash((sid, h, q)) % 2**31).random() * 10,
                    "era5": np.random.default_rng(hash((sid, h, q, "e")) % 2**31).random() * 10,
                    "observation_type": obs_type,
                    "lat": 40.0,
                    "lon": -100.0,
                    "elevation_m": 300.0,
                    "slope_deg": 2.0,
                    "aspect_deg": 180.0,
                    "neighbors_10km_site_ids": "ASOS001" if obs_type == "GS" else "",
                })
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_wide_quantile_df(qcols):
    """Wide-format DataFrame: 3 sites with metadata + q000..q100."""
    rows = []
    for i, sid in enumerate(["A001", "A002", "A003"]):
        row = {
            "station_id": sid,
            "height_m": 60,
            "lat": 35.0 + i,
            "lon": -100.0 + i,
        }
        row.update(dict(zip(qcols, np.linspace(0, 5 + i, 101))))
        rows.append(row)
    return pd.DataFrame(rows)
