"""Tests for GWA interpolation helpers (moved from loocv_xgb to utils.ml)."""

import numpy as np
import pandas as pd

from wem.utils.ml import _interp_gwa_row


class TestInterpGwaRow:
    def test_precomputed_present(self):
        row = pd.Series({"gwa_interp": 6.5, "height_m": 60})
        result = _interp_gwa_row(row, 60.0, "gwa_interp")
        assert result == 6.5

    def test_power_law_interpolation(self):
        row = pd.Series({
            "gwa_10": 4.0,
            "gwa_50": 6.0,
            "gwa_100": 7.5,
            "gwa_150": 8.5,
            "height_m": 80,
        })
        result = _interp_gwa_row(row, 80.0, "gwa_interp")
        assert result is not None
        assert 6.0 < result < 7.5

    def test_insufficient_data(self):
        row = pd.Series({"gwa_10": 4.0, "height_m": 60})
        result = _interp_gwa_row(row, 60.0, "gwa_interp")
        assert result is None

    def test_nan_values(self):
        row = pd.Series({
            "gwa_interp": np.nan,
            "gwa_10": np.nan,
            "gwa_50": np.nan,
            "height_m": 60,
        })
        result = _interp_gwa_row(row, 60.0, "gwa_interp")
        assert result is None

    def test_negative_value_fallthrough(self):
        row = pd.Series({"gwa_interp": -1.0, "height_m": 60})
        result = _interp_gwa_row(row, 60.0, "gwa_interp")
        # -1.0 is not > 0, so falls through to power law
        assert result is None  # No gwa_* columns to fall back on

    def test_zero_target_height(self):
        row = pd.Series({
            "gwa_10": 4.0,
            "gwa_50": 6.0,
            "gwa_100": 7.5,
            "height_m": 0,
        })
        result = _interp_gwa_row(row, 0.0, "gwa_interp")
        assert result is None

    def test_precomputed_zero(self):
        row = pd.Series({"gwa_interp": 0.0, "height_m": 60})
        result = _interp_gwa_row(row, 60.0, "gwa_interp")
        # 0.0 is not > 0, falls through
        assert result is None

    def test_inf_precomputed(self):
        row = pd.Series({"gwa_interp": np.inf, "height_m": 60})
        result = _interp_gwa_row(row, 60.0, "gwa_interp")
        assert result is None
