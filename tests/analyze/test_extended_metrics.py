"""Tests for wem.analyze.extended_metrics — pure function tests with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wem.analyze.extended_metrics import (
    aggregate_site_means,
    compute_summary_metrics,
    compute_height_metrics,
)


# ---- aggregate_site_means ---------------------------------------------------

def test_aggregate_site_means_basic():
    """101 quantile rows -> 1 aggregated site row with mean and bias columns."""
    n = 101
    df = pd.DataFrame({
        "station_id": ["S1"] * n,
        "observation_type": ["GS"] * n,
        "height_m": [80.0] * n,
        "qnum": list(range(n)),
        "observation": np.linspace(1.0, 101.0, n),
        "era5": np.linspace(1.5, 101.5, n),
    })

    result = aggregate_site_means(df, dataset_cols=["era5"])

    assert len(result) == 1
    assert "mean_observation" in result.columns
    assert "mean_era5" in result.columns
    assert "bias_era5" in result.columns
    # ERA5 values are uniformly 0.5 higher, so bias should be ~0.5
    assert abs(result["bias_era5"].iloc[0] - 0.5) < 0.1


def test_aggregate_site_means_min_qrows():
    """Fewer rows than min_qrows -> empty output."""
    df = pd.DataFrame({
        "station_id": ["S1"] * 5,
        "observation_type": ["GS"] * 5,
        "height_m": [80.0] * 5,
        "qnum": list(range(5)),
        "observation": [1.0, 2.0, 3.0, 4.0, 5.0],
        "era5": [1.5, 2.5, 3.5, 4.5, 5.5],
    })

    result = aggregate_site_means(df, dataset_cols=["era5"], min_qrows=10)
    assert len(result) == 0


def test_aggregate_site_means_no_dataset_cols():
    """No matching dataset columns -> ValueError."""
    df = pd.DataFrame({
        "station_id": ["S1"],
        "observation_type": ["GS"],
        "height_m": [80.0],
        "qnum": [0],
        "observation": [5.0],
    })

    with pytest.raises(ValueError, match="No requested dataset columns"):
        aggregate_site_means(df, dataset_cols=["nonexistent"])


# ---- compute_summary_metrics ------------------------------------------------

def test_compute_summary_metrics_basic():
    """Single-row site df -> summary with correct dataset, subset, median_bias."""
    site = pd.DataFrame({
        "station_id": ["S1"],
        "observation_type": ["GS"],
        "height_m": [80.0],
        "mean_observation": [5.0],
        "bias_era5": [0.5],
    })

    result = compute_summary_metrics(site, dataset_keys=["era5"])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["dataset"] == "era5"
    assert row["subset"] == "GS"
    assert abs(row["median_bias"] - 0.5) < 1e-9


# ---- compute_height_metrics -------------------------------------------------

def test_compute_height_metrics_basic():
    """Single GS site at 80 m -> height metric row with mean_bias."""
    site = pd.DataFrame({
        "station_id": ["S1"],
        "observation_type": ["GS"],
        "height_m": [80.0],
        "mean_observation": [5.0],
        "bias_era5": [0.3],
    })

    result = compute_height_metrics(site, dataset_keys=["era5"])

    assert len(result) == 1
    assert "height_band_m" in result.columns
    assert "mean_bias" in result.columns
    assert abs(result["mean_bias"].iloc[0] - 0.3) < 1e-9


def test_compute_height_metrics_empty():
    """Empty site df -> empty output with correct columns."""
    site = pd.DataFrame(columns=[
        "station_id", "observation_type", "height_m",
        "mean_observation", "bias_era5",
    ])

    result = compute_height_metrics(site, dataset_keys=["era5"])

    assert len(result) == 0
    expected_cols = {
        "dataset", "subset", "height_band_m", "mean_bias",
        "mean_abs_bias", "median_abs_bias", "mean_abs_pct_bias", "n_sites",
    }
    assert expected_cols == set(result.columns)
