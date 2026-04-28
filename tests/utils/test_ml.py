"""Tests for wem.utils.ml."""

import numpy as np
import pandas as pd
import pytest

from wem.utils.ml import (
    balance_indices,
    build_neighbor_map,
    fold_seed,
    make_features,
    pick_present,
)


# ---- pick_present ----

class TestPickPresent:
    def test_all_exist(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        assert pick_present(df, ["a", "b"]) == ["a", "b"]

    def test_some_missing(self):
        df = pd.DataFrame(columns=["a", "c"])
        assert pick_present(df, ["a", "b", "c"]) == ["a", "c"]

    def test_none_exist(self):
        df = pd.DataFrame(columns=["x", "y"])
        assert pick_present(df, ["a", "b"]) == []

    def test_order_preserved(self):
        df = pd.DataFrame(columns=["c", "b", "a"])
        assert pick_present(df, ["a", "b", "c"]) == ["a", "b", "c"]


# ---- make_features ----

class TestMakeFeatures:
    def test_is_gs_flag(self):
        df = pd.DataFrame({
            "observation_type": ["GS", "ASOS", "GS"],
            "lat": [40, 41, 42],
        })
        result = make_features(df)
        assert list(result["is_gs"]) == [1, 0, 1]
        assert result["is_gs"].dtype == np.int8

    def test_aspect_trig(self):
        df = pd.DataFrame({
            "observation_type": ["ASOS"],
            "aspect_deg": [90.0],
        })
        result = make_features(df)
        assert "aspect_sin" in result.columns
        assert "aspect_cos" in result.columns
        assert abs(result["aspect_sin"].iloc[0] - 1.0) < 0.01
        assert abs(result["aspect_cos"].iloc[0] - 0.0) < 0.01

    def test_no_aspect_no_trig(self):
        df = pd.DataFrame({"observation_type": ["ASOS"], "lat": [40]})
        result = make_features(df)
        assert "aspect_sin" not in result.columns
        assert "aspect_cos" not in result.columns

    def test_returns_copy(self):
        df = pd.DataFrame({"observation_type": ["GS"]})
        result = make_features(df)
        assert result is not df

    def test_various_obs_types(self):
        df = pd.DataFrame({
            "observation_type": ["gs", "goldstandard", "gold standard", "ASOS", "unknown"],
        })
        result = make_features(df)
        expected = [1, 1, 1, 0, 0]
        assert list(result["is_gs"]) == expected


# ---- build_neighbor_map ----

class TestBuildNeighborMap:
    def test_basic_parse(self):
        df = pd.DataFrame({
            "station_id": ["GS1", "GS1", "A1"],
            "observation_type": ["GS", "GS", "ASOS"],
            "neighbors_10km_site_ids": ["A1,A2", "A1,A2", ""],
        })
        nbr = build_neighbor_map(df)
        assert "GS1" in nbr
        assert nbr["GS1"] == {"A1", "A2"}

    def test_empty_neighbors(self):
        df = pd.DataFrame({
            "station_id": ["GS1"],
            "observation_type": ["GS"],
            "neighbors_10km_site_ids": [""],
        })
        nbr = build_neighbor_map(df)
        assert nbr["GS1"] == set()

    def test_no_gs_sites(self):
        df = pd.DataFrame({
            "station_id": ["A1", "A2"],
            "observation_type": ["ASOS", "ASOS"],
            "neighbors_10km_site_ids": ["A2", "A1"],
        })
        nbr = build_neighbor_map(df)
        assert nbr == {}

    def test_missing_column(self):
        df = pd.DataFrame({
            "station_id": ["GS1"],
            "observation_type": ["GS"],
        })
        nbr = build_neighbor_map(df)
        assert nbr == {}


# ---- balance_indices ----

class TestBalanceIndices:
    def test_downsample(self):
        rng = np.random.default_rng(42)
        idx_asos = np.arange(100)
        idx_gs = np.arange(100, 130)
        result = balance_indices(idx_asos, idx_gs, rng, "downsample")
        assert len(result) == 60  # 30 + 30

    def test_upsample(self):
        rng = np.random.default_rng(42)
        idx_asos = np.arange(100)
        idx_gs = np.arange(100, 130)
        result = balance_indices(idx_asos, idx_gs, rng, "upsample")
        assert len(result) == 200  # 100 + 100

    def test_one_side_empty(self):
        rng = np.random.default_rng(42)
        idx_asos = np.arange(50)
        idx_gs = np.array([], dtype=int)
        result = balance_indices(idx_asos, idx_gs, rng)
        assert len(result) == 50

    def test_equal_sizes(self):
        rng = np.random.default_rng(42)
        idx_asos = np.arange(30)
        idx_gs = np.arange(30, 60)
        result = balance_indices(idx_asos, idx_gs, rng)
        assert len(result) == 60  # 30 + 30

    def test_deterministic_with_seed(self):
        idx_a = np.arange(100)
        idx_g = np.arange(100, 130)
        r1 = balance_indices(idx_a, idx_g, np.random.default_rng(42))
        r2 = balance_indices(idx_a, idx_g, np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)


# ---- fold_seed ----

class TestFoldSeed:
    def test_deterministic(self):
        s1 = fold_seed(42, "station_A")
        s2 = fold_seed(42, "station_A")
        assert s1 == s2

    def test_different_sids(self):
        s1 = fold_seed(42, "station_A")
        s2 = fold_seed(42, "station_B")
        assert s1 != s2

    def test_within_32bit(self):
        s = fold_seed(42, "test_station")
        assert 0 <= s < 2**32

    def test_different_base_seeds(self):
        s1 = fold_seed(1, "station_X")
        s2 = fold_seed(2, "station_X")
        assert s1 != s2
