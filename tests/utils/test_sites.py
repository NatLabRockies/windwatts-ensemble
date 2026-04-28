"""Tests for wem.utils.sites."""

import pandas as pd
import pytest

from wem.utils.sites import (
    already_done,
    already_done_gs,
    load_gs_sites,
    load_sites,
    normalize_obs_type,
)


# ---- normalize_obs_type ----

class TestNormalizeObsType:
    @pytest.mark.parametrize("inp,expected", [
        ("gs", "GS"),
        ("GS", "GS"),
        ("goldstandard", "GS"),
        ("gold standard", "GS"),
        ("gold_stand", "GS"),
        ("gold-std", "GS"),
        ("gold", "GS"),
        ("ASOS", "ASOS"),
        ("asos", "ASOS"),
    ])
    def test_known_variants(self, inp, expected):
        assert normalize_obs_type(inp) == expected

    def test_unknown(self):
        result = normalize_obs_type("buoy")
        assert result == "buoy"

    def test_whitespace(self):
        assert normalize_obs_type("  GS  ") == "GS"

    def test_non_string(self):
        assert normalize_obs_type(42) == ""

    def test_empty_string(self):
        result = normalize_obs_type("")
        assert result == ""


# ---- load_sites ----

class TestLoadSites:
    def test_standard_columns(self, tmp_path):
        p = tmp_path / "sites.csv"
        p.write_text("station_id,name,lat,lon,elev_m\nA001,Site A,40.0,-100.0,300\n")
        df = load_sites(p)
        assert len(df) == 1
        assert df["station_id"].iloc[0] == "A001"
        assert df["lat"].iloc[0] == 40.0

    def test_flexible_names(self, tmp_path):
        p = tmp_path / "sites.csv"
        # load_sites needs station_id/site_id + lat + lon as minimum
        p.write_text("STATION,name,Latitude,Longitude,elevation_m\nB001,test,35.0,-90.0,500\n")
        df = load_sites(p)
        assert len(df) == 1
        assert df["lat"].iloc[0] == 35.0

    def test_drops_nan_latlon(self, tmp_path):
        p = tmp_path / "sites.csv"
        p.write_text("station_id,name,lat,lon,elev_m\nA,s1,40,-100,10\nB,s2,,-80,20\n")
        df = load_sites(p)
        assert len(df) == 1
        assert df["station_id"].iloc[0] == "A"

    def test_deduplicates(self, tmp_path):
        p = tmp_path / "sites.csv"
        p.write_text(
            "station_id,name,lat,lon,elev_m\n"
            "A,s1,40,-100,10\n"
            "A,s1,40,-100,10\n"
        )
        df = load_sites(p)
        assert len(df) == 1

    def test_numeric_conversion(self, tmp_path):
        p = tmp_path / "sites.csv"
        p.write_text("station_id,name,lat,lon,elev_m\nA,s,40.5,-99.5,300.1\n")
        df = load_sites(p)
        assert df["lat"].dtype == float
        assert df["lon"].dtype == float
        assert df["elev_m"].dtype == float

    def test_missing_required_raises(self, tmp_path):
        p = tmp_path / "sites.csv"
        p.write_text("x,y\n1,2\n")
        with pytest.raises(ValueError, match="station_id"):
            load_sites(p)


# ---- already_done ----

class TestAlreadyDone:
    def test_existing_file(self, tmp_path):
        p = tmp_path / "out.csv"
        p.write_text("station_id,value\nA001,1\nA002,2\n")
        result = already_done(p)
        assert result == {"A001", "A002"}

    def test_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.csv"
        result = already_done(p)
        assert result == set()

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("station_id\n")
        result = already_done(p)
        assert result == set()

    def test_corrupted(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("garbage\nno headers\n")
        result = already_done(p)
        assert result == set()


# ---- load_gs_sites ----

class TestLoadGsSites:
    def test_standard_columns(self, tmp_path):
        p = tmp_path / "gs.csv"
        p.write_text(
            "station_id,name,lat,lon,elev_m,height_m\n"
            "GS01,Site1,40.0,-100.0,300,60\n"
            "GS01,Site1,40.0,-100.0,300,80\n"
        )
        df = load_gs_sites(p)
        assert len(df) == 2
        assert set(df["height_m"]) == {60.0, 80.0}

    def test_dedup_by_station_height(self, tmp_path):
        p = tmp_path / "gs.csv"
        p.write_text(
            "station_id,name,lat,lon,elev_m,height_m\n"
            "GS01,Site1,40.0,-100.0,300,60\n"
            "GS01,Site1,40.0,-100.0,300,60\n"
        )
        df = load_gs_sites(p)
        assert len(df) == 1

    def test_drops_nan_height(self, tmp_path):
        p = tmp_path / "gs.csv"
        p.write_text(
            "station_id,name,lat,lon,elev_m,height_m\n"
            "GS01,s,40.0,-100.0,300,60\n"
            "GS02,s,41.0,-101.0,300,\n"
        )
        df = load_gs_sites(p)
        assert len(df) == 1
        assert df["station_id"].iloc[0] == "GS01"

    def test_missing_height_col_raises(self, tmp_path):
        p = tmp_path / "gs.csv"
        p.write_text("station_id,lat,lon\nA,40,-100\n")
        with pytest.raises(ValueError, match="height_m"):
            load_gs_sites(p)

    def test_flexible_column_names(self, tmp_path):
        p = tmp_path / "gs.csv"
        p.write_text("site_id,NAME,LAT,LON,elevation_m,z\nX,s,40,-100,300,80\n")
        df = load_gs_sites(p)
        assert len(df) == 1
        assert df["height_m"].iloc[0] == 80.0


# ---- already_done_gs ----

class TestAlreadyDoneGs:
    def test_existing_file(self, tmp_path):
        p = tmp_path / "out.csv"
        p.write_text("station_id,height_m,value\nGS01,60.0,1\nGS01,80.0,2\n")
        result = already_done_gs(p)
        assert result == {("GS01", 60.0), ("GS01", 80.0)}

    def test_missing_file(self, tmp_path):
        p = tmp_path / "missing.csv"
        result = already_done_gs(p)
        assert result == set()

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("station_id,height_m\n")
        result = already_done_gs(p)
        assert result == set()

    def test_returns_tuples(self, tmp_path):
        p = tmp_path / "out.csv"
        p.write_text("station_id,height_m\nA,10.0\n")
        result = already_done_gs(p)
        assert isinstance(result, set)
        item = next(iter(result))
        assert isinstance(item, tuple)
        assert item == ("A", 10.0)
