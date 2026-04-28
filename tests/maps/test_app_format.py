"""Tests for wem.maps.app_format."""

import numpy as np
import pandas as pd
import pytest

from wem.maps.app_format import (
    count_adjacent_equals,
    enforce_monotonic,
    normalize_index_series,
    power_alpha,
    to_output_profile,
    vertical_fill_one,
)


# ---- normalize_index_series ----

class TestNormalizeIndexSeries:
    def test_zero_pads(self):
        s = pd.Series([1, 23, 456])
        result = normalize_index_series(s)
        assert list(result) == ["000001", "000023", "000456"]

    def test_string_input(self):
        s = pd.Series(["123", "00456"])
        result = normalize_index_series(s)
        assert list(result) == ["000123", "000456"]

    def test_float_string(self):
        s = pd.Series(["123.0"])
        result = normalize_index_series(s)
        assert result.iloc[0] == "000123"

    def test_too_many_digits_raises(self):
        s = pd.Series([1234567])
        with pytest.raises(ValueError, match="6 digits"):
            normalize_index_series(s)

    def test_nan_raises(self):
        s = pd.Series([np.nan])
        with pytest.raises(ValueError, match="Missing"):
            normalize_index_series(s)


# ---- enforce_monotonic ----

class TestEnforceMonotonic:
    def test_already_monotonic(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        result = enforce_monotonic(arr)
        np.testing.assert_array_equal(result, arr)

    def test_fixes_dip(self):
        arr = np.array([0.0, 2.0, 1.0, 3.0])
        result = enforce_monotonic(arr)
        # After np.maximum.accumulate: [0, 2, 2, 3]
        expected = np.array([0.0, 2.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_handles_nan(self):
        arr = np.array([0.0, np.nan, 2.0])
        result = enforce_monotonic(arr)
        assert np.all(np.isfinite(result))
        assert result[-1] == 2.0

    def test_all_zeros(self):
        arr = np.zeros(10)
        result = enforce_monotonic(arr)
        np.testing.assert_array_equal(result, arr)


# ---- power_alpha ----

class TestPowerAlpha:
    def test_known_shear(self):
        # v2 = v1 * (h2/h1)^alpha => alpha = ln(v2/v1) / ln(h2/h1)
        # 5 * (100/10)^0.14 ≈ 6.90
        v1 = np.array([5.0])
        v2 = np.array([6.9])
        a = power_alpha(v1, v2, 10.0, 100.0)
        assert abs(a[0] - 0.14) < 0.02

    def test_equal_speeds(self):
        v = np.array([5.0])
        a = power_alpha(v, v, 10.0, 100.0)
        assert abs(a[0]) < 1e-6

    def test_clamps_extreme(self):
        v1 = np.array([0.001])
        v2 = np.array([1000.0])
        a = power_alpha(v1, v2, 10.0, 100.0)
        assert a[0] <= 2.0


# ---- vertical_fill_one ----

class TestVerticalFillOne:
    def test_exact_height_exists(self):
        have = {60: np.array([3.0, 4.0, 5.0])}
        result = vertical_fill_one(60, have, 101)
        np.testing.assert_array_equal(result, have[60])

    def test_interpolates_between(self):
        have = {
            40: np.array([3.0, 4.0, 5.0]),
            100: np.array([5.0, 6.0, 7.0]),
        }
        result = vertical_fill_one(60, have, 101)
        assert result is not None
        # Should be between 40m and 100m values
        assert np.all(result >= 3.0)
        assert np.all(result <= 7.0)

    def test_extrapolates_single_neighbor(self):
        have = {80: np.array([5.0, 6.0])}
        result = vertical_fill_one(100, have, 101)
        assert result is not None
        # Power law with default alpha=1/7, h_ratio=100/80=1.25
        assert np.all(result > 5.0)

    def test_empty_returns_none(self):
        result = vertical_fill_one(60, {}, 101)
        assert result is None


# ---- to_output_profile ----

class TestToOutputProfile:
    def test_101_output(self):
        q101 = np.linspace(0, 10, 101)
        result = to_output_profile(q101, 101)
        assert len(result) == 101
        assert result[0] == 0.0
        assert result[-1] == 10.0

    def test_33_output(self):
        q101 = np.linspace(0, 10, 101)
        result = to_output_profile(q101, 33)
        assert len(result) == 33
        assert abs(result[0]) < 0.01
        assert abs(result[-1] - 10.0) < 0.01

    def test_enforces_monotonicity(self):
        q101 = np.linspace(0, 10, 101)
        q101[50] = 0.0  # big dip
        result = to_output_profile(q101, 101)
        diffs = np.diff(result)
        assert np.all(diffs >= 0)


# ---- count_adjacent_equals ----

class TestCountAdjacentEquals:
    def test_no_equals(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert count_adjacent_equals(arr) == 0

    def test_with_equals(self):
        arr = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        assert count_adjacent_equals(arr) == 2

    def test_all_equal(self):
        arr = np.array([5.0, 5.0, 5.0])
        assert count_adjacent_equals(arr) == 2

    def test_with_tolerance(self):
        arr = np.array([1.0, 1.001, 2.0])
        assert count_adjacent_equals(arr, tol=0.01) == 1
        assert count_adjacent_equals(arr, tol=0.0) == 0
