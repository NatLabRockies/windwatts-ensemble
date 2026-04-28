"""Tests for wem.assemble.obs_wind_avg."""

import numpy as np
import pandas as pd
import pytest

from wem.assemble.obs_wind_avg import first_nonnull, mean_from_qnum_block


# ---- mean_from_qnum_block ----

class TestMeanFromQnumBlock:
    def test_linear(self):
        qnum = np.arange(101, dtype=float)
        obs = np.linspace(0, 10, 101)
        result = mean_from_qnum_block(qnum, obs)
        assert abs(result - 5.0) < 0.1

    def test_constant(self):
        qnum = np.arange(101, dtype=float)
        obs = np.full(101, 3.0)
        result = mean_from_qnum_block(qnum, obs)
        assert abs(result - 3.0) < 0.01

    def test_too_few(self):
        qnum = np.array([0.0])
        obs = np.array([5.0])
        result = mean_from_qnum_block(qnum, obs)
        assert np.isnan(result)

    def test_nan_handling(self):
        qnum = np.arange(101, dtype=float)
        obs = np.linspace(0, 10, 101)
        obs[10] = np.nan
        obs[50] = np.nan
        result = mean_from_qnum_block(qnum, obs)
        # Should still produce a value (NaN points dropped)
        assert np.isfinite(result)

    def test_unsorted_qnum(self):
        qnum = np.array([100, 0, 50], dtype=float)
        obs = np.array([10.0, 0.0, 5.0])
        result = mean_from_qnum_block(qnum, obs)
        assert abs(result - 5.0) < 0.5

    def test_qmax_zero(self):
        qnum = np.array([0.0, 0.0])
        obs = np.array([1.0, 2.0])
        result = mean_from_qnum_block(qnum, obs)
        assert np.isnan(result)


# ---- first_nonnull ----

class TestFirstNonnull:
    def test_basic(self):
        s = pd.Series([np.nan, np.nan, 5.0, 6.0])
        assert first_nonnull(s) == 5.0

    def test_all_nan(self):
        s = pd.Series([np.nan, np.nan])
        assert np.isnan(first_nonnull(s))

    def test_first_valid(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert first_nonnull(s) == 1.0
