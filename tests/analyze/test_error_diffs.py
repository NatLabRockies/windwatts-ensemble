"""Tests for wem.analyze.error_diffs — pure function tests with synthetic data."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from wem.analyze.error_diffs import compute_group_diffs, main


def _make_df(station_ids, obs, pred, era5, height_m=80.0):
    """Helper to build a DataFrame for compute_group_diffs tests."""
    n = len(station_ids)
    return pd.DataFrame({
        "station_id": station_ids,
        "height_m": [height_m] * n,
        "observation": obs,
        "pred_observation": pred,
        "era5": era5,
    })


# ---- compute_group_diffs ----------------------------------------------------

def test_compute_group_diffs_ml_better():
    """ML closer to obs than dataset -> diff < 0."""
    # obs=5, pred=5.1 (|err|=0.1), era5=7 (|err|=2)
    df = _make_df(
        station_ids=["S1"] * 5,
        obs=[5.0] * 5,
        pred=[5.1] * 5,
        era5=[7.0] * 5,
    )
    result = compute_group_diffs(df, dataset_col="era5")
    assert len(result) == 1
    assert result["diff"].iloc[0] < 0.0


def test_compute_group_diffs_ml_worse():
    """ML farther from obs than dataset -> diff > 0."""
    # obs=5, pred=9 (|err|=4), era5=5.1 (|err|=0.1)
    df = _make_df(
        station_ids=["S1"] * 5,
        obs=[5.0] * 5,
        pred=[9.0] * 5,
        era5=[5.1] * 5,
    )
    result = compute_group_diffs(df, dataset_col="era5")
    assert len(result) == 1
    assert result["diff"].iloc[0] > 0.0


def test_compute_group_diffs_equal():
    """ML and dataset have same error -> diff = 0."""
    # obs=5, pred=6 (|err|=1), era5=4 (|err|=1)
    df = _make_df(
        station_ids=["S1"] * 5,
        obs=[5.0] * 5,
        pred=[6.0] * 5,
        era5=[4.0] * 5,
    )
    result = compute_group_diffs(df, dataset_col="era5")
    assert len(result) == 1
    assert abs(result["diff"].iloc[0]) < 1e-9


def test_compute_group_diffs_sorted():
    """Output is sorted from most negative to most positive diff."""
    # S1: obs=5, pred=5.1, era5=8 -> diff very negative (ML much better)
    # S2: obs=5, pred=8,   era5=5.1 -> diff very positive (ML much worse)
    # S3: obs=5, pred=6,   era5=4 -> diff ~0
    df = pd.DataFrame({
        "station_id": ["S1"] * 3 + ["S2"] * 3 + ["S3"] * 3,
        "height_m": [80.0] * 9,
        "observation": [5.0] * 9,
        "pred_observation": [5.1] * 3 + [8.0] * 3 + [6.0] * 3,
        "era5": [8.0] * 3 + [5.1] * 3 + [4.0] * 3,
    })
    result = compute_group_diffs(df, dataset_col="era5")
    diffs = result["diff"].tolist()
    assert diffs == sorted(diffs)


def test_compute_group_diffs_multiple_groups():
    """Three station_ids -> 3 output rows."""
    rows = []
    for sid in ["A", "B", "C"]:
        for _ in range(5):
            rows.append({
                "station_id": sid,
                "height_m": 80.0,
                "observation": 5.0,
                "pred_observation": 5.5,
                "era5": 6.0,
            })
    df = pd.DataFrame(rows)
    result = compute_group_diffs(df, dataset_col="era5")
    assert len(result) == 3


# ---- main exists -------------------------------------------------------------

def test_main_exists():
    assert callable(main)
