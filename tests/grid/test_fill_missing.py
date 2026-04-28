"""Tests for wem.grid.fill_missing."""

import numpy as np
import pandas as pd
import pytest

from wem.grid.fill_missing import process_df


class TestProcessDf:
    def test_nan_to_zero(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        result = process_df(df, numeric_only=False, zero_str=False, replace_inf=True)
        assert result["a"].iloc[1] == 0
        assert result["b"].iloc[0] == 0

    def test_empty_string_to_zero(self):
        df = pd.DataFrame({"a": ["hello", "", "world"]})
        result = process_df(df, numeric_only=False, zero_str=False, replace_inf=True)
        assert result["a"].iloc[1] == 0

    def test_inf_handling(self):
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf]})
        result = process_df(df, numeric_only=False, zero_str=False, replace_inf=True)
        assert result["a"].iloc[1] == 0
        assert result["a"].iloc[2] == 0

    def test_non_nan_preserved(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = process_df(df, numeric_only=False, zero_str=False, replace_inf=True)
        pd.testing.assert_frame_equal(df, result)

    def test_numeric_only(self):
        df = pd.DataFrame({"num": [1.0, np.nan], "str_col": ["a", None]})
        result = process_df(df, numeric_only=True, zero_str=False, replace_inf=True)
        assert result["num"].iloc[1] == 0
        # str_col should still be NaN (we didn't set zero_str)
        assert pd.isna(result["str_col"].iloc[1])

    def test_numeric_only_with_zero_str(self):
        df = pd.DataFrame({"num": [1.0, np.nan], "str_col": ["a", None]})
        result = process_df(df, numeric_only=True, zero_str=True, replace_inf=True)
        assert result["num"].iloc[1] == 0
        assert result["str_col"].iloc[1] == "0"

    def test_no_inf_fix(self):
        df = pd.DataFrame({"a": [1.0, np.inf]})
        result = process_df(df, numeric_only=False, zero_str=False, replace_inf=False)
        assert result["a"].iloc[1] == np.inf

    def test_mixed_types(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, np.nan, 3.0],
            "str_col": ["a", "b", "c"],
        })
        result = process_df(df, numeric_only=False, zero_str=False, replace_inf=True)
        assert result["float_col"].iloc[1] == 0
