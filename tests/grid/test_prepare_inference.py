"""Tests for wem.grid.prepare_inference."""

import numpy as np
import pandas as pd
import pytest

from wem.grid.prepare_inference import (
    find_dataset_column,
    find_quantile_columns,
    normalize_dataset_names,
)


# ---- normalize_dataset_names ----

class TestNormalizeDatasetNames:
    @pytest.mark.parametrize("inp,expected", [
        ("WTK CONUS", "wtk"),
        ("HRRR CONUS", "hrrr"),
        ("WTK-LED CONUS", "wtk_led_conus"),
        ("wtk", "wtk"),
        ("hrrr", "hrrr"),
        ("wtk_led_conus", "wtk_led_conus"),
    ])
    def test_known(self, inp, expected):
        s = pd.Series([inp])
        result = normalize_dataset_names(s)
        assert result.iloc[0] == expected

    def test_case_insensitive(self):
        s = pd.Series(["WTK conus", "Hrrr CONUS"])
        result = normalize_dataset_names(s)
        assert result.iloc[0] == "wtk"
        assert result.iloc[1] == "hrrr"

    def test_passthrough(self):
        s = pd.Series(["unknown_dataset"])
        result = normalize_dataset_names(s)
        assert result.iloc[0] == "unknown_dataset"


# ---- find_dataset_column ----

class TestFindDatasetColumn:
    def test_dataset_found(self):
        df = pd.DataFrame(columns=["lat", "lon", "dataset", "q000"])
        assert find_dataset_column(df) == "dataset"

    def test_source_found(self):
        df = pd.DataFrame(columns=["lat", "lon", "source", "q000"])
        assert find_dataset_column(df) == "source"

    def test_missing_raises(self):
        df = pd.DataFrame(columns=["lat", "lon", "q000"])
        with pytest.raises(ValueError, match="Could not find"):
            find_dataset_column(df)

    def test_priority_order(self):
        df = pd.DataFrame(columns=["dataset", "source", "model"])
        assert find_dataset_column(df) == "dataset"


# ---- find_quantile_columns ----

class TestFindQuantileColumns:
    def test_standard(self):
        cols = ["lat", "lon"] + [f"q{i:03d}" for i in range(101)]
        result = find_quantile_columns(cols)
        assert len(result) == 101

    def test_sorted(self):
        cols = ["q100", "q000", "q050"]
        result = find_quantile_columns(cols)
        assert result == ["q000", "q050", "q100"]

    def test_no_match_raises(self):
        cols = ["lat", "lon", "value"]
        with pytest.raises(ValueError, match="No quantile columns"):
            find_quantile_columns(cols)

    def test_4digit(self):
        cols = [f"q{i:04d}" for i in range(0, 1001, 10)]
        result = find_quantile_columns(cols)
        assert len(result) == 101
