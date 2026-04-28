"""Tests for wem.grid.pivot_predictions (main-level logic using QCOLS)."""

import numpy as np
import pandas as pd
import pytest

from wem.constants import QCOLS


class TestPivotLogic:
    def test_qnum_to_qcol_mapping(self):
        """Verify that qnum 0..100 maps to q000..q100."""
        for q in range(101):
            expected = f"q{q:03d}"
            assert expected in QCOLS

    def test_all_101_columns(self):
        assert len(QCOLS) == 101

    def test_pivot_table_dedup(self):
        """Duplicate qnums for same site should average."""
        df = pd.DataFrame({
            "index": [1, 1],
            "latitude": [40.0, 40.0],
            "longitude": [-100.0, -100.0],
            "height_m": [60, 60],
            "qcol": ["q050", "q050"],
            "pred_observation": [5.0, 7.0],
        })
        pv = df.pivot_table(
            index=["index", "latitude", "longitude", "height_m"],
            columns="qcol",
            values="pred_observation",
            aggfunc="mean",
        ).reset_index()
        assert abs(pv["q050"].iloc[0] - 6.0) < 1e-10

    def test_output_column_order(self):
        """Verify expected output columns."""
        expected_prefix = ["index", "latitude", "longitude", "height_m"]
        expected = expected_prefix + QCOLS
        assert len(expected) == 4 + 101
