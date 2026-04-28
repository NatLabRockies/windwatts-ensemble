"""Tests for wem.utils.quantiles."""

import numpy as np
import pandas as pd
import pytest

from wem.utils.quantiles import (
    mean_from_quantile_long,
    mean_from_quantiles,
    mean_from_quantiles_row,
    quantile_block,
)


class TestQuantileBlock:
    def test_shape(self):
        spd = np.random.default_rng(0).random((1000, 3)).astype("float32")
        result = quantile_block(spd)
        assert result.shape == (101, 3)

    def test_monotonic(self):
        spd = np.random.default_rng(1).random((500, 2))
        result = quantile_block(spd)
        for col in range(result.shape[1]):
            assert np.all(np.diff(result[:, col]) >= 0)

    def test_min_max(self):
        rng = np.random.default_rng(2)
        spd = rng.random((200, 1)) * 15
        result = quantile_block(spd)
        assert np.isclose(result[0, 0], np.nanmin(spd), atol=1e-5)
        assert np.isclose(result[100, 0], np.nanmax(spd), atol=1e-5)

    def test_constant_input(self):
        spd = np.full((100, 1), 5.0)
        result = quantile_block(spd)
        assert np.allclose(result, 5.0)

    def test_nan_handling(self):
        spd = np.array([[1, 2, np.nan, 4, 5]] * 100).T.reshape(5, -1)
        result = quantile_block(spd)
        # Should produce valid quantiles (nanpercentile ignores NaN)
        assert result.shape[0] == 101
        assert np.all(np.isfinite(result))


class TestMeanFromQuantiles:
    def test_linear_ramp(self, qcols):
        vals = np.linspace(0, 10, 101)
        df = pd.DataFrame([dict(zip(qcols, vals))])
        result = mean_from_quantiles(df)
        assert abs(result[0] - 5.0) < 0.1

    def test_constant(self, qcols):
        vals = np.full(101, 3.5)
        df = pd.DataFrame([dict(zip(qcols, vals))])
        result = mean_from_quantiles(df)
        assert abs(result[0] - 3.5) < 0.01

    def test_dtype_float32(self, qcols):
        vals = np.linspace(0, 5, 101)
        df = pd.DataFrame([dict(zip(qcols, vals))])
        result = mean_from_quantiles(df)
        assert result.dtype == np.float32

    def test_multi_row(self, qcols):
        rows = []
        for scale in [2.0, 4.0, 8.0]:
            rows.append(dict(zip(qcols, np.linspace(0, scale, 101))))
        df = pd.DataFrame(rows)
        result = mean_from_quantiles(df)
        assert len(result) == 3
        assert abs(result[0] - 1.0) < 0.1
        assert abs(result[1] - 2.0) < 0.1
        assert abs(result[2] - 4.0) < 0.1


class TestMeanFromQuantilesRow:
    def test_linear(self):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {c: np.linspace(0, 10, 101)[i] for i, c in enumerate(qcols)}
        row = pd.Series(data)
        result = mean_from_quantiles_row(row, qcols)
        assert result is not None
        assert abs(result - 5.0) < 0.1

    def test_constant(self):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {c: 3.0 for c in qcols}
        row = pd.Series(data)
        result = mean_from_quantiles_row(row, qcols)
        assert result is not None
        assert abs(result - 3.0) < 0.01

    def test_insufficient_data(self):
        row = pd.Series({"q000": 5.0})
        result = mean_from_quantiles_row(row, ["q000"])
        assert result is None

    def test_nan_handling(self):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {c: np.linspace(0, 10, 101)[i] for i, c in enumerate(qcols)}
        data["q050"] = np.nan
        row = pd.Series(data)
        result = mean_from_quantiles_row(row, qcols)
        assert result is not None
        assert np.isfinite(result)

    def test_all_nan(self):
        qcols = [f"q{i:03d}" for i in range(3)]
        data = {c: np.nan for c in qcols}
        row = pd.Series(data)
        result = mean_from_quantiles_row(row, qcols)
        assert result is None


class TestMeanFromQuantileLong:
    def test_linear(self):
        qnum = np.arange(101, dtype=float)
        values = np.linspace(0, 10, 101)
        result = mean_from_quantile_long(qnum, values)
        assert abs(result - 5.0) < 0.1

    def test_constant(self):
        qnum = np.arange(101, dtype=float)
        values = np.full(101, 3.0)
        result = mean_from_quantile_long(qnum, values)
        assert abs(result - 3.0) < 0.1

    def test_too_few(self):
        qnum = np.array([0.0])
        values = np.array([5.0])
        result = mean_from_quantile_long(qnum, values)
        assert np.isnan(result)

    def test_nan_filtered(self):
        qnum = np.arange(101, dtype=float)
        values = np.linspace(0, 10, 101)
        values[50] = np.nan
        result = mean_from_quantile_long(qnum, values)
        assert np.isfinite(result)

    def test_all_nan(self):
        qnum = np.array([0.0, 50.0, 100.0])
        values = np.array([np.nan, np.nan, np.nan])
        result = mean_from_quantile_long(qnum, values)
        assert np.isnan(result)

    def test_unsorted_input(self):
        qnum = np.arange(101, dtype=float)
        values = np.linspace(0, 10, 101)
        # Shuffle both arrays the same way
        idx = np.random.default_rng(0).permutation(101)
        result = mean_from_quantile_long(qnum[idx], values[idx])
        assert abs(result - 5.0) < 0.1
