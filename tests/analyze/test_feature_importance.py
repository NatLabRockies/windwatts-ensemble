"""Tests for wem.analyze.feature_importance: nice_name, read_fi_table, write_normalized_summary."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wem.analyze.feature_importance import (
    nice_name,
    read_fi_table,
    write_normalized_summary,
)
from wem.constants import FEATURE_DISPLAY_MAP


class TestNiceName:
    def test_nice_name_mapped(self):
        assert nice_name("era5") == "ERA5"

    def test_nice_name_unmapped(self):
        assert nice_name("foo") == "foo"

    def test_nice_name_uses_constants(self):
        assert nice_name("wtk_led_conus") == "WTK-LED CONUS"
        # Confirm it actually comes from the constants map
        assert FEATURE_DISPLAY_MAP["wtk_led_conus"] == "WTK-LED CONUS"


class TestReadFiTable:
    def test_read_fi_table_basic(self, tmp_path: Path):
        csv = tmp_path / "fi.csv"
        df = pd.DataFrame(
            {
                "feature": ["era5", "hrrr", "gwa_interp"],
                "weight": [100, 80, 60],
                "gain": [0.5, 0.3, 0.2],
                "cover": [500.0, 400.0, 300.0],
            }
        )
        df.to_csv(csv, index=False)

        result = read_fi_table(csv)
        assert result.shape[0] == 3
        assert "feature_disp" in result.columns
        # Check the display names are applied
        assert result.loc[result["feature"] == "era5", "feature_disp"].iloc[0] == "ERA5"

    def test_read_fi_table_missing_col(self, tmp_path: Path):
        csv = tmp_path / "fi_bad.csv"
        df = pd.DataFrame(
            {
                "feature": ["era5"],
                "weight": [100],
                # 'gain' is intentionally missing
                "cover": [500.0],
            }
        )
        df.to_csv(csv, index=False)

        with pytest.raises(ValueError, match="Missing required column 'gain'"):
            read_fi_table(csv)


class TestWriteNormalizedSummary:
    def test_write_normalized_summary(self, tmp_path: Path):
        out_csv = tmp_path / "summary.csv"
        df = pd.DataFrame(
            {
                "feature": ["era5", "hrrr"],
                "feature_disp": ["ERA5", "HRRR"],
                "weight": [100, 50],
                "gain": [0.8, 0.2],
                "cover": [600.0, 400.0],
                "total_gain": [80.0, 10.0],
                "total_cover": [60000.0, 20000.0],
            }
        )
        write_normalized_summary(df, out_csv)

        result = pd.read_csv(out_csv)
        assert "gain_share" in result.columns
        assert "gain_rank" in result.columns
        # gain_share should sum to 1.0
        np.testing.assert_allclose(result["gain_share"].sum(), 1.0, atol=1e-10)
        # The row with higher gain should have rank 1
        top = result.loc[result["gain_rank"] == 1]
        assert top["feature"].iloc[0] == "era5"
