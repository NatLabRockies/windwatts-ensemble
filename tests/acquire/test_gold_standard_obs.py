"""Tests for wem.acquire.gold_standard_obs."""

import numpy as np
import pandas as pd
import pytest

from wem.acquire.gold_standard_obs import (
    compute_group_quantiles,
    month_balance_filter_table,
    normalize_columns,
)


# ---- normalize_columns ----

class TestNormalizeColumns:
    def test_standard_names(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "ws_observed": [5.0, 6.0, 7.0],
            "site_id": ["A", "A", "A"],
            "height": [60, 60, 60],
            "lat": [40.0, 40.0, 40.0],
            "lon": [-100.0, -100.0, -100.0],
        })
        result = normalize_columns(df)
        assert "ws_observed" in result.columns
        assert "site_id" in result.columns
        assert result["ws_observed"].dtype == float

    def test_flexible_names(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=2, freq="h"),
            "speed": [5.0, 6.0],
            "station_id": ["B", "B"],
            "z": [80, 80],
            "latitude": [35.0, 35.0],
            "longitude": [-90.0, -90.0],
        })
        result = normalize_columns(df)
        assert "ws_observed" in result.columns
        assert len(result) == 2

    def test_missing_required_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="Missing required"):
            normalize_columns(df)

    def test_negative_ws_to_nan(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "ws_observed": [-1.0, 5.0, -0.5],
            "site_id": ["A", "A", "A"],
            "height": [60, 60, 60],
            "lat": [40.0, 40.0, 40.0],
            "lon": [-100.0, -100.0, -100.0],
        })
        result = normalize_columns(df)
        assert np.isnan(result["ws_observed"].iloc[0])
        assert result["ws_observed"].iloc[1] == 5.0

    def test_datetime_parsing(self):
        df = pd.DataFrame({
            "datetime": ["2020-01-01 00:00", "2020-01-01 01:00"],
            "ws_observed": [5.0, 6.0],
            "site_id": ["A", "A"],
            "height": [60, 60],
            "lat": [40.0, 40.0],
            "lon": [-100.0, -100.0],
        })
        result = normalize_columns(df)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])


# ---- compute_group_quantiles ----

class TestComputeGroupQuantiles:
    def test_basic(self):
        s = pd.Series(np.linspace(0, 10, 1000))
        result = compute_group_quantiles(s)
        assert len(result) == 101
        assert result["q000"] < result["q100"]

    def test_empty(self):
        s = pd.Series([], dtype=float)
        result = compute_group_quantiles(s)
        assert len(result) == 101
        assert all(np.isnan(v) for v in result.values)

    def test_with_nan(self):
        s = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        result = compute_group_quantiles(s)
        assert np.isfinite(result["q000"])
        assert np.isfinite(result["q100"])

    def test_101_keys(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = compute_group_quantiles(s)
        expected_keys = [f"q{p:03d}" for p in range(101)]
        assert list(result.index) == expected_keys


# ---- month_balance_filter_table ----

class TestMonthBalanceFilterTable:
    def _make_df(self, months, counts_per_month):
        """Helper: create a df with samples spread across months."""
        rows = []
        for m, count in zip(months, counts_per_month):
            for _ in range(count):
                rows.append({
                    "datetime": pd.Timestamp(year=2020, month=m, day=1),
                    "site_id": "S1",
                    "height": 60,
                    "ws_observed": 5.0,
                })
        return pd.DataFrame(rows)

    def test_all_12_months_kept(self):
        df = self._make_df(range(1, 13), [100] * 12)
        result = month_balance_filter_table(df, 2020, 2024, min_months=12, min_frac_of_median=0.5)
        assert result["kept"].iloc[0] == True

    def test_few_months_rejected(self):
        df = self._make_df([1, 2, 3], [100, 100, 100])
        result = month_balance_filter_table(df, 2020, 2024, min_months=12, min_frac_of_median=0.5)
        assert result["kept"].iloc[0] == False

    def test_low_counts_rejected(self):
        # 12 months present, but some have very low counts
        counts = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1]
        df = self._make_df(range(1, 13), counts)
        result = month_balance_filter_table(df, 2020, 2024, min_months=12, min_frac_of_median=0.5)
        # Median of nonzero counts = 100, threshold = 50; month 12 has 1 < 50
        # So months_ge_threshold = 11, which < 12 -> rejected
        assert result["kept"].iloc[0] == False

    def test_reason_column(self):
        df = self._make_df([1, 2], [100, 100])
        result = month_balance_filter_table(df, 2020, 2024, min_months=12, min_frac_of_median=0.5)
        assert "reason" in result.columns
        assert result["reason"].iloc[0] != "ok"
