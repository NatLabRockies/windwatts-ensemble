"""Tests for wem.analyze.ml_results."""

import numpy as np
import pandas as pd
import pytest

from wem.analyze.ml_results import mean_from_quantile_series
from wem.utils.plotting import robust_limits, symmetric_bias_limit


# ---- mean_from_quantile_series ----

class TestMeanFromQuantileSeries:
    def test_linear(self):
        qnum = np.arange(101, dtype=float)
        values = np.linspace(0, 10, 101)
        result = mean_from_quantile_series(qnum, values)
        assert abs(result - 5.0) < 0.1

    def test_constant(self):
        qnum = np.arange(101, dtype=float)
        values = np.full(101, 3.0)
        result = mean_from_quantile_series(qnum, values)
        assert abs(result - 3.0) < 0.1

    def test_too_few(self):
        qnum = np.array([0.0])
        values = np.array([5.0])
        result = mean_from_quantile_series(qnum, values)
        assert np.isnan(result)

    def test_nan_filtered(self):
        qnum = np.arange(101, dtype=float)
        values = np.linspace(0, 10, 101)
        values[50] = np.nan
        result = mean_from_quantile_series(qnum, values)
        assert np.isfinite(result)

    def test_all_nan(self):
        qnum = np.array([0.0, 50.0, 100.0])
        values = np.array([np.nan, np.nan, np.nan])
        result = mean_from_quantile_series(qnum, values)
        assert np.isnan(result)


# ---- robust_limits ----

class TestRobustLimits:
    def test_known_distribution(self):
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(0, 1, 10000))
        lo, hi = robust_limits([s], trim=0.02)
        assert lo < 0
        assert hi > 0
        assert abs(lo) < 3
        assert abs(hi) < 3

    def test_empty(self):
        lo, hi = robust_limits([pd.Series([], dtype=float)])
        assert lo == 0.0
        assert hi == 1.0

    def test_custom_trim(self):
        s = pd.Series(np.linspace(-10, 10, 1000))
        lo, hi = robust_limits([s], trim=0.1)
        # 10th and 90th percentile of uniform -10..10 -> ~-8, ~8
        assert abs(lo - (-8)) < 0.5
        assert abs(hi - 8) < 0.5


# ---- symmetric_bias_limit ----

class TestSymmetricBiasLimit:
    def test_symmetric(self):
        s = pd.Series(np.linspace(-5, 5, 1000))
        L = symmetric_bias_limit([s], trim=0.02)
        assert L > 0
        assert abs(L - 5.0) < 0.5

    def test_empty(self):
        L = symmetric_bias_limit([pd.Series([], dtype=float)], trim=0.02)
        assert L == 5.0  # default fallback

    def test_one_sided(self):
        s = pd.Series(np.linspace(0, 10, 1000))
        L = symmetric_bias_limit([s], trim=0.02)
        assert L > 0
