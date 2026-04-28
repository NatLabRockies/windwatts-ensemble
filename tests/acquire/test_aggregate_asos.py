"""Tests for wem.acquire.aggregate_asos."""

import json

import numpy as np
import pandas as pd
import pytest

from wem.acquire.aggregate_asos import (
    build_row,
    read_header_and_table,
    years_from_header,
)


# ---- read_header_and_table ----

class TestReadHeaderAndTable:
    def test_valid(self, tmp_path):
        p = tmp_path / "test_quantiles.csv"
        header = {"station_id": "A001", "years_used": "2010,2011,2012"}
        p.write_text(
            "# " + json.dumps(header) + "\n"
            "quantile,wind_speed_m_s\n"
            "0,1.0\n"
            "50,5.0\n"
            "100,10.0\n"
        )
        h, df = read_header_and_table(p)
        assert h["station_id"] == "A001"
        assert len(df) == 3
        assert "quantile" in df.columns
        assert "wind_speed_m_s" in df.columns

    def test_missing_header_prefix(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("quantile,wind_speed_m_s\n0,1.0\n")
        with pytest.raises(ValueError, match="Missing JSON header"):
            read_header_and_table(p)

    def test_missing_columns(self, tmp_path):
        p = tmp_path / "bad2.csv"
        header = {"station_id": "A"}
        p.write_text("# " + json.dumps(header) + "\na,b\n1,2\n")
        with pytest.raises(ValueError, match="Missing required"):
            read_header_and_table(p)


# ---- years_from_header ----

class TestYearsFromHeader:
    def test_valid(self):
        h = {"years_used": "2010,2011,2012,2013,2014"}
        result = years_from_header(h)
        assert result == [2010, 2011, 2012, 2013, 2014]

    def test_empty(self):
        h = {}
        result = years_from_header(h)
        assert result == []

    def test_garbage(self):
        h = {"years_used": "abc,def,ghi"}
        result = years_from_header(h)
        assert result == []

    def test_mixed(self):
        h = {"years_used": "2020,bad,2021"}
        result = years_from_header(h)
        assert result == [2020, 2021]


# ---- build_row ----

class TestBuildRow:
    def test_all_quantiles_present(self):
        header = {"station_id": "A001", "years_used": "2020"}
        qtab = pd.DataFrame({
            "quantile": list(range(101)),
            "wind_speed_m_s": np.linspace(0, 10, 101),
        })
        row = build_row(header, qtab)
        for q in range(101):
            key = f"q{q:03d}"
            assert key in row
            assert row[key] is not None

    def test_missing_quantiles_interpolated(self):
        header = {"station_id": "A002"}
        # Only have q0, q50, q100
        qtab = pd.DataFrame({
            "quantile": [0, 50, 100],
            "wind_speed_m_s": [0.0, 5.0, 10.0],
        })
        row = build_row(header, qtab)
        # q025 should be interpolated to ~2.5
        assert row["q025"] is not None
        assert abs(row["q025"] - 2.5) < 0.1

    def test_metadata_preserved(self):
        header = {"station_id": "A003", "STATE": "TX", "nested": {"key": "val"}}
        qtab = pd.DataFrame({
            "quantile": list(range(101)),
            "wind_speed_m_s": np.linspace(0, 5, 101),
        })
        row = build_row(header, qtab)
        assert row["station_id"] == "A003"
        assert row["STATE"] == "TX"
        # Nested dict gets serialized to JSON string
        assert '"key"' in row["nested"]
