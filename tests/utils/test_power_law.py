"""Tests for wem.utils.power_law."""

import numpy as np
import pytest

from wem.utils.power_law import bracket_for_height, fit_power_law_alpha, power_law_interp


# ---- bracket_for_height ----

class TestBracketForHeight:
    def test_exact_match(self):
        avail = np.array([10, 40, 60, 80, 100])
        is_exact, lo, hi = bracket_for_height(60, avail)
        assert is_exact is True
        assert lo == hi == 60

    def test_between(self):
        avail = np.array([10, 40, 60, 80, 100])
        is_exact, lo, hi = bracket_for_height(50, avail)
        assert is_exact is False
        assert lo == 40
        assert hi == 60

    def test_below_min(self):
        avail = np.array([10, 40, 60, 80, 100])
        is_exact, lo, hi = bracket_for_height(5, avail)
        assert is_exact is False
        assert lo == 10
        assert hi == 40

    def test_above_max(self):
        avail = np.array([10, 40, 60, 80, 100])
        is_exact, lo, hi = bracket_for_height(150, avail)
        assert is_exact is False
        assert lo == 80
        assert hi == 100

    def test_float_noise_tolerance(self):
        avail = np.array([10, 40, 60, 80, 100])
        is_exact, lo, hi = bracket_for_height(60.0000005, avail)
        assert is_exact is True
        assert lo == hi == 60

    def test_single_element(self):
        avail = np.array([100])
        is_exact, lo, hi = bracket_for_height(100, avail)
        assert is_exact is True


# ---- power_law_interp ----

class TestPowerLawInterp:
    def test_exact_height(self):
        hv = [(10, 3.0), (40, 5.0), (100, 7.0)]
        assert power_law_interp(40, hv) == 5.0

    def test_between_two_heights(self):
        hv = [(10, 3.0), (100, 6.0)]
        result = power_law_interp(50, hv)
        assert 3.0 < result < 6.0

    def test_below_range(self):
        hv = [(40, 4.0), (100, 7.0)]
        result = power_law_interp(10, hv)
        assert np.isfinite(result)
        assert result < 4.0

    def test_above_range(self):
        hv = [(10, 3.0), (40, 5.0)]
        result = power_law_interp(100, hv)
        assert np.isfinite(result)
        assert result > 5.0

    def test_single_value(self):
        hv = [(40, 5.0)]
        result = power_law_interp(100, hv)
        assert result == 5.0

    def test_empty_returns_nan(self):
        result = power_law_interp(50, [])
        assert np.isnan(result)

    def test_nan_filtering(self):
        hv = [(10, np.nan), (40, 5.0), (np.nan, 3.0)]
        result = power_law_interp(40, hv)
        assert result == 5.0

    def test_zero_speed_linear_fallback(self):
        hv = [(10, 0.0), (100, 5.0)]
        result = power_law_interp(50, hv)
        # Linear interpolation between 0 and 5
        assert 0 < result < 5

    def test_power_law_consistency(self):
        # If U = A * z^alpha, with known alpha=0.3
        alpha = 0.3
        A = 2.0
        hv = [(10, A * 10**alpha), (100, A * 100**alpha)]
        result = power_law_interp(50, hv)
        expected = A * 50**alpha
        assert abs(result - expected) < 0.01


# ---- fit_power_law_alpha ----

class TestFitPowerLawAlpha:
    def test_two_points(self):
        heights = np.array([10, 100])
        speeds = np.array([3.0, 6.0])
        result = fit_power_law_alpha(heights, speeds)
        assert result is not None
        A, alpha = result
        assert A > 0
        assert 0 < alpha < 1

    def test_known_alpha_recovery(self):
        alpha_true = 0.3
        A_true = 2.0
        heights = np.array([10, 40, 80, 100, 200])
        speeds = A_true * heights.astype(float) ** alpha_true
        result = fit_power_law_alpha(heights, speeds)
        assert result is not None
        A, alpha = result
        assert abs(alpha - alpha_true) < 0.01
        assert abs(A - A_true) < 0.1

    def test_fewer_than_2_points(self):
        result = fit_power_law_alpha(np.array([10]), np.array([5.0]))
        assert result is None

    def test_all_nan(self):
        result = fit_power_law_alpha(
            np.array([10, 20, 30]),
            np.array([np.nan, np.nan, np.nan]),
        )
        assert result is None

    def test_negative_filtering(self):
        heights = np.array([10, 20, 30, 40])
        speeds = np.array([-1, 0, 5.0, 7.0])
        result = fit_power_law_alpha(heights, speeds)
        # Only 2 valid points: (30, 5.0) and (40, 7.0)
        assert result is not None

    def test_zero_height_filtered(self):
        heights = np.array([0, 10, 20])
        speeds = np.array([1.0, 3.0, 5.0])
        result = fit_power_law_alpha(heights, speeds)
        assert result is not None

    def test_all_zero_speeds(self):
        result = fit_power_law_alpha(np.array([10, 20]), np.array([0.0, 0.0]))
        assert result is None
