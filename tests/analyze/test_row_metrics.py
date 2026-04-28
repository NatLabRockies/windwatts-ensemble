"""Tests for wem.analyze.row_metrics — pure function tests with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wem.analyze.row_metrics import compute_row_metrics


def _make_df(obs, era5, obs_type="GS", n=20):
    """Helper to build a simple quantile DataFrame."""
    return pd.DataFrame({
        "station_id": ["S1"] * n,
        "observation_type": [obs_type] * n,
        "observation": obs,
        "era5": era5,
        "qnum": list(range(n)),
    })


# ---- compute_row_metrics ----------------------------------------------------

def test_compute_row_metrics_perfect():
    """When era5 == observation, RMSE/MAE/bias should all be 0."""
    n = 20
    vals = np.linspace(1.0, 10.0, n)
    df = _make_df(obs=vals, era5=vals, n=n)

    result = compute_row_metrics(df, obs_col="observation", dataset_cols=["era5"], subset="GS")

    assert len(result) == 1
    row = result.iloc[0]
    assert abs(row["rmse_row"]) < 1e-9
    assert abs(row["mae_row"]) < 1e-9
    assert abs(row["bias_row"]) < 1e-9


def test_compute_row_metrics_known_bias():
    """When era5 = observation + 1.0, bias=1.0 and mae=1.0."""
    n = 20
    obs = np.linspace(1.0, 10.0, n)
    era5 = obs + 1.0
    df = _make_df(obs=obs, era5=era5, n=n)

    result = compute_row_metrics(df, obs_col="observation", dataset_cols=["era5"], subset="GS")

    row = result.iloc[0]
    assert abs(row["bias_row"] - 1.0) < 1e-9
    assert abs(row["mae_row"] - 1.0) < 1e-9


def test_compute_row_metrics_gs_filter():
    """Only GS rows are used when subset='GS'."""
    n_gs = 10
    n_asos = 10
    obs_gs = np.ones(n_gs) * 5.0
    era5_gs = np.ones(n_gs) * 6.0  # bias = +1
    obs_asos = np.ones(n_asos) * 5.0
    era5_asos = np.ones(n_asos) * 10.0  # bias = +5 (should be ignored)

    df = pd.DataFrame({
        "station_id": ["S1"] * n_gs + ["S2"] * n_asos,
        "observation_type": ["GS"] * n_gs + ["ASOS"] * n_asos,
        "observation": np.concatenate([obs_gs, obs_asos]),
        "era5": np.concatenate([era5_gs, era5_asos]),
        "qnum": list(range(n_gs)) + list(range(n_asos)),
    })

    result = compute_row_metrics(df, obs_col="observation", dataset_cols=["era5"], subset="GS")

    row = result.iloc[0]
    # Should only see bias from the GS rows (+1), not ASOS (+5)
    assert abs(row["bias_row"] - 1.0) < 1e-9
    assert row["n_rows"] == n_gs


def test_compute_row_metrics_no_cols():
    """dataset_cols=['nonexistent'] -> ValueError."""
    df = _make_df(obs=[1.0], era5=[1.0], n=1)

    with pytest.raises(ValueError, match="None of the requested dataset columns"):
        compute_row_metrics(df, obs_col="observation", dataset_cols=["nonexistent"], subset="GS")


def test_compute_row_metrics_missing_obs_col():
    """obs_col='missing' -> ValueError."""
    df = _make_df(obs=[1.0], era5=[1.0], n=1)

    with pytest.raises(ValueError, match="Input missing required column"):
        compute_row_metrics(df, obs_col="missing", dataset_cols=["era5"], subset="GS")


def test_compute_row_metrics_returns_bias_row():
    """Verify bias_row column is in the output."""
    n = 20
    vals = np.linspace(1.0, 10.0, n)
    df = _make_df(obs=vals, era5=vals + 0.5, n=n)

    result = compute_row_metrics(df, obs_col="observation", dataset_cols=["era5"], subset="GS")

    assert "bias_row" in result.columns
