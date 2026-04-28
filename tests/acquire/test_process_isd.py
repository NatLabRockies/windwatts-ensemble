"""Tests for wem.acquire.process_isd."""

import numpy as np
import pandas as pd
import pytest

from wem.acquire.process_isd import (
    days_in_year,
    dequantize_ceil,
    expected_per_day,
    make_quantiles,
    parse_speed_ms,
    years_with_enough_data,
)


# ---- days_in_year ----

class TestDaysInYear:
    @pytest.mark.parametrize("year,expected", [
        (2023, 365),
        (2024, 366),
        (1900, 365),
        (2000, 366),
        (2100, 365),
        (1996, 366),
    ])
    def test_values(self, year, expected):
        assert days_in_year(year) == expected


# ---- parse_speed_ms ----

class TestParseSpeedMs:
    def test_valid_wnd(self):
        s = pd.Series(["090,5,N,50,5"])  # parts[3]=50 -> 50/10=5.0
        result = parse_speed_ms(s)
        assert result.iloc[0] == 5.0

    def test_9999_to_nan(self):
        s = pd.Series(["090,5,N,9999,5"])
        result = parse_speed_ms(s)
        assert np.isnan(result.iloc[0])

    def test_malformed(self):
        s = pd.Series(["garbage"])
        result = parse_speed_ms(s)
        assert np.isnan(result.iloc[0])

    def test_too_few_parts(self):
        s = pd.Series(["090,5"])
        result = parse_speed_ms(s)
        assert np.isnan(result.iloc[0])


# ---- expected_per_day ----

class TestExpectedPerDay:
    def test_hourly(self):
        dates = pd.date_range("2020-01-01", periods=48, freq="h")
        df = pd.DataFrame({"DATE": dates})
        assert expected_per_day(df) == 24

    def test_empty(self):
        df = pd.DataFrame({"DATE": pd.Series([], dtype="datetime64[ns]")})
        assert expected_per_day(df) == 0


# ---- years_with_enough_data ----

class TestYearsWithEnoughData:
    def test_pass(self):
        # 24 samples/day, 365 days -> 8760 expected, 95% = 8322
        dates = pd.date_range("2020-01-01", periods=24 * 365, freq="h")
        df = pd.DataFrame({"DATE": dates})
        result = years_with_enough_data(df, 24)
        assert 2020 in result

    def test_fail(self):
        # Only 100 samples in a year
        dates = pd.date_range("2020-01-01", periods=100, freq="h")
        df = pd.DataFrame({"DATE": dates})
        result = years_with_enough_data(df, 24)
        assert 2020 not in result

    def test_boundary_95(self):
        # 2021 is not a leap year -> 365 days -> expected = 24*365 = 8760
        n_expected = 24 * 365
        n_samples = int(0.95 * n_expected)
        dates = pd.date_range("2021-01-01", periods=n_samples, freq="h")
        df = pd.DataFrame({"DATE": dates})
        result = years_with_enough_data(df, 24)
        assert 2021 in result


# ---- dequantize_ceil ----

class TestDequantizeCeil:
    def test_calm(self):
        s = pd.Series([0.0, 0.0, 0.0])
        result = dequantize_ceil(s)
        # Calm: U(0, 2] kt -> 0 to ~1.03 m/s
        assert all(result >= 0)
        assert all(result <= 2.0 * 0.514444 + 0.01)

    def test_nonzero_knot(self):
        # 10 kt ~ 5.14 m/s
        s = pd.Series([5.14444])
        result = dequantize_ceil(s)
        assert result.iloc[0] >= 0

    def test_nan_preservation(self):
        s = pd.Series([np.nan, 5.0, np.nan])
        result = dequantize_ceil(s)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[2])
        assert result.iloc[1] >= 0

    def test_output_nonnegative(self):
        s = pd.Series([0.0, 1.0, 3.0, 5.0, 10.0])
        result = dequantize_ceil(s)
        assert all(result[np.isfinite(result)] >= 0)

    def test_different_seeds_produce_different_results(self):
        s = pd.Series([0.0] * 100)
        np.random.seed(1)
        r1 = dequantize_ceil(s).copy()
        np.random.seed(2)
        r2 = dequantize_ceil(s).copy()
        # Different seeds should give different random samples
        assert not np.allclose(r1, r2)


# ---- make_quantiles ----

class TestMakeQuantiles:
    def test_shape(self):
        speed = pd.Series(np.random.default_rng(0).random(1000) * 10)
        result = make_quantiles(speed)
        assert len(result) == 101

    def test_range(self):
        speed = pd.Series(np.random.default_rng(1).random(500) * 10)
        result = make_quantiles(speed)
        assert result["quantile"].min() == 0
        assert result["quantile"].max() == 100

    def test_monotonic(self):
        speed = pd.Series(np.random.default_rng(2).random(1000) * 10)
        result = make_quantiles(speed)
        vals = result["wind_speed_m_s"].values
        assert all(np.diff(vals) >= -1e-10)
