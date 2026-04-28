"""Tests for wem.maps.diff_maps."""

import numpy as np
import pandas as pd
import pytest

from wem.maps.diff_maps import (
    merge_for_diff,
    round_coords,
    unique_prediction_heights,
)


# ---- round_coords ----

class TestRoundCoords:
    def test_rounds(self):
        df = pd.DataFrame({"lat": [40.1234], "lon": [-100.5678]})
        result = round_coords(df, 2)
        assert result["lat_r"].iloc[0] == 40.12
        assert result["lon_r"].iloc[0] == -100.57

    def test_preserves_original(self):
        df = pd.DataFrame({"lat": [40.1234], "lon": [-100.5678]})
        result = round_coords(df, 2)
        assert result["lat"].iloc[0] == 40.1234


# ---- unique_prediction_heights ----

class TestUniquePredictionHeights:
    def test_returns_sorted(self):
        df = pd.DataFrame({"height_m": [100, 60, 80, 60, 100]})
        result = unique_prediction_heights(df)
        assert result == [60, 80, 100]


# ---- merge_for_diff ----

class TestMergeForDiff:
    def test_basic_merge(self):
        pred = pd.DataFrame({
            "lat": [40.0], "lon": [-100.0], "mean_ws": [5.0],
            "lat_r": [40.0], "lon_r": [-100.0],
        })
        ds = pd.DataFrame({
            "lat": [40.0], "lon": [-100.0], "mean_ws": [4.5],
            "lat_r": [40.0], "lon_r": [-100.0],
        })
        result = merge_for_diff(pred, ds)
        assert len(result) == 1
        assert abs(result["pred"].iloc[0] - 5.0) < 0.01
        assert abs(result["ds"].iloc[0] - 4.5) < 0.01

    def test_empty_input(self):
        pred = pd.DataFrame(columns=["lat", "lon", "mean_ws", "lat_r", "lon_r"])
        ds = pd.DataFrame({
            "lat": [40.0], "lon": [-100.0], "mean_ws": [4.5],
            "lat_r": [40.0], "lon_r": [-100.0],
        })
        result = merge_for_diff(pred, ds)
        assert result.empty

    def test_no_overlap(self):
        pred = pd.DataFrame({
            "lat": [40.0], "lon": [-100.0], "mean_ws": [5.0],
            "lat_r": [40.0], "lon_r": [-100.0],
        })
        ds = pd.DataFrame({
            "lat": [50.0], "lon": [-90.0], "mean_ws": [4.5],
            "lat_r": [50.0], "lon_r": [-90.0],
        })
        result = merge_for_diff(pred, ds)
        assert result.empty
