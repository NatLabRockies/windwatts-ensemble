"""Tests for wem.utils.columns."""

import pandas as pd
import pytest

from wem.utils.columns import choose_col, find_qcols


# ---- choose_col ----

class TestChooseCol:
    def test_exact_match(self):
        df = pd.DataFrame(columns=["lat", "lon", "elev"])
        assert choose_col(df, ["lat", "latitude"]) == "lat"

    def test_case_insensitive_fallback(self):
        df = pd.DataFrame(columns=["LAT", "LON"])
        assert choose_col(df, ["lat"]) == "LAT"

    def test_no_match(self):
        df = pd.DataFrame(columns=["x", "y"])
        assert choose_col(df, ["lat", "lon"]) is None

    def test_priority_order(self):
        df = pd.DataFrame(columns=["station_id", "site_id"])
        assert choose_col(df, ["station_id", "site_id"]) == "station_id"

    def test_empty_candidates(self):
        df = pd.DataFrame(columns=["a", "b"])
        assert choose_col(df, []) is None

    def test_empty_df(self):
        df = pd.DataFrame()
        assert choose_col(df, ["a"]) is None

    def test_case_insensitive_priority(self):
        df = pd.DataFrame(columns=["Lat", "LAT"])
        # Exact match for "Lat" should win
        assert choose_col(df, ["Lat"]) == "Lat"


# ---- find_qcols ----

class TestFindQcols:
    def test_full_101(self, qcols):
        df = pd.DataFrame(columns=["station_id"] + qcols)
        result = find_qcols(df)
        assert len(result) == 101
        assert result[0] == "q000"
        assert result[-1] == "q100"

    def test_partial_above_50(self, qcols):
        subset = qcols[:60]
        df = pd.DataFrame(columns=["id"] + subset)
        result = find_qcols(df)
        assert len(result) == 60

    def test_partial_below_50_regex_fallback(self):
        cols = [f"q{i:03d}" for i in range(30)]
        df = pd.DataFrame(columns=["x"] + cols)
        result = find_qcols(df)
        assert len(result) == 30
        assert result == sorted(cols)

    def test_no_match(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        result = find_qcols(df)
        assert result == []

    def test_regex_catches_noncanonical(self):
        # Below 50 canonical triggers regex; 4-char q+digits match
        cols = [f"q{i:03d}" for i in range(10)]
        df = pd.DataFrame(columns=cols)
        result = find_qcols(df)
        assert len(result) == 10

    def test_threshold_at_50(self, qcols):
        subset = qcols[:50]
        df = pd.DataFrame(columns=subset)
        result = find_qcols(df)
        assert len(result) == 50

    def test_non_q_columns_excluded(self):
        cols = [f"q{i:03d}" for i in range(5)] + ["quantity", "quality"]
        df = pd.DataFrame(columns=cols)
        result = find_qcols(df)
        # Below threshold: regex fallback. "quantity" has 8 chars, "quality" has 7 chars -> excluded
        assert "quantity" not in result
        assert "quality" not in result
