"""Tests for wem.train.final_xgb."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.requires_xgboost

try:
    import xgboost  # noqa: F401
except Exception as exc:
    pytest.skip(f"xgboost unavailable: {exc}", allow_module_level=True)

from wem.utils.ml import balance_indices
from wem.train.final_xgb import metrics


# ---- balance_indices (downsample mode, formerly balance_downsample) ----

class TestBalanceDownsample:
    def test_basic(self):
        rng = np.random.default_rng(42)
        idx_a = np.arange(100)
        idx_b = np.arange(100, 130)
        result = balance_indices(idx_a, idx_b, rng, strategy="downsample")
        assert len(result) == 60  # 30 + 30

    def test_equal_sizes(self):
        rng = np.random.default_rng(42)
        idx_a = np.arange(50)
        idx_b = np.arange(50, 100)
        result = balance_indices(idx_a, idx_b, rng, strategy="downsample")
        assert len(result) == 100

    def test_one_side_empty(self):
        rng = np.random.default_rng(42)
        idx_a = np.arange(50)
        idx_b = np.array([], dtype=int)
        result = balance_indices(idx_a, idx_b, rng, strategy="downsample")
        assert len(result) == 50
        np.testing.assert_array_equal(result, idx_a)

    def test_other_side_empty(self):
        rng = np.random.default_rng(42)
        idx_a = np.array([], dtype=int)
        idx_b = np.arange(30)
        result = balance_indices(idx_a, idx_b, rng, strategy="downsample")
        assert len(result) == 30

    def test_deterministic(self):
        idx_a = np.arange(100)
        idx_b = np.arange(100, 130)
        r1 = balance_indices(idx_a, idx_b, np.random.default_rng(42), strategy="downsample")
        r2 = balance_indices(idx_a, idx_b, np.random.default_rng(42), strategy="downsample")
        np.testing.assert_array_equal(r1, r2)


# ---- metrics ----

class TestMetrics:
    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = metrics(y_true, y_pred)
        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0
        assert result["n"] == 4

    def test_all_nan(self):
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([1.0, 2.0])
        result = metrics(y_true, y_pred)
        assert np.isnan(result["rmse"])
        assert result["n"] == 0

    def test_perfect_prediction(self):
        y_true = np.array([5.0, 10.0, 15.0])
        y_pred = np.array([5.0, 10.0, 15.0])
        result = metrics(y_true, y_pred)
        assert result["rmse"] == 0.0
        assert result["mae"] == 0.0

    def test_known_error(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        result = metrics(y_true, y_pred)
        assert abs(result["rmse"] - 1.0) < 1e-10
        assert abs(result["mae"] - 1.0) < 1e-10
        assert result["n"] == 2

    def test_mixed_finite(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = metrics(y_true, y_pred)
        assert result["n"] == 2
        assert result["rmse"] == 0.0
