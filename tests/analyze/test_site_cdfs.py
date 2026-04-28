"""Tests for wem.analyze.site_cdfs — pure function tests with synthetic data."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from wem.analyze.site_cdfs import validate_columns, compute_xlim, plot_site_cdf, main


# ---- validate_columns -------------------------------------------------------

def test_validate_columns_all_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    missing = validate_columns(df, ["a", "b", "c"])
    assert missing == []


def test_validate_columns_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    missing = validate_columns(df, ["a", "foo"])
    assert missing == ["foo"]


# ---- compute_xlim -----------------------------------------------------------

def test_compute_xlim_basic():
    df = pd.DataFrame({"col": [0.0, 5.0, 10.0, 15.0, 20.0]})
    xmin, xmax = compute_xlim(df, ["col"])
    assert xmin == 0.0
    expected_99th = float(np.nanpercentile([0, 5, 10, 15, 20], 99)) * 1.05
    assert abs(xmax - expected_99th) < 1e-6


def test_compute_xlim_all_nan():
    df = pd.DataFrame({"col": [np.nan, np.nan, np.nan]})
    xmin, xmax = compute_xlim(df, ["col"])
    assert xmin == 0.0
    assert xmax == 20.0


# ---- plot_site_cdf ----------------------------------------------------------

def test_plot_site_cdf_creates_file(tmp_path):
    """Build a tiny group DataFrame and verify that plot_site_cdf writes a PNG."""
    n = 101
    qnums = list(range(n))
    vals = np.linspace(0.0, 15.0, n)

    group = pd.DataFrame({
        "station_id": ["ST001"] * n,
        "qnum": qnums,
        "observation": vals,
        "era5": vals + 0.1,
        "wtk": vals + 0.2,
        "hrrr": vals - 0.1,
        "wtk_led_conus": vals + 0.3,
        "wtk_led_climate": vals + 0.05,
        "height_m": [80.0] * n,
        "name": ["TestSite"] * n,
    })

    xlim = (0.0, 16.0)
    plot_site_cdf(group, xlim, tmp_path)

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 0


# ---- main exists -------------------------------------------------------------

def test_main_exists():
    assert callable(main)
