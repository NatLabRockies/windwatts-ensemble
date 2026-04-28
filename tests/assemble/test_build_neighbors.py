"""Tests for wem.assemble.build_neighbors."""

import numpy as np
import pandas as pd
import pytest

from wem.assemble.build_neighbors import build_neighbor_lists


class TestBuildNeighborLists:
    def test_nearby_mutual(self):
        # Two GS sites ~0 km apart (same location), one ASOS nearby
        sites = pd.DataFrame({
            "station_id": ["GS1", "GS2", "A1"],
            "lat": [40.0, 40.0001, 40.0],
            "lon": [-100.0, -100.0001, -100.0002],
            "observation_type": ["GS", "GS", "ASOS"],
        })
        result = build_neighbor_lists(sites, radius_km=5.0)
        # GS1 should have GS2 and A1 as neighbors
        gs1_row = result[result["station_id"] == "GS1"]
        assert not gs1_row.empty
        nbrs = gs1_row["neighbors_10km_site_ids"].iloc[0].split(",")
        assert "GS2" in nbrs
        assert "A1" in nbrs

    def test_far_no_neighbors(self):
        sites = pd.DataFrame({
            "station_id": ["GS1", "A1"],
            "lat": [40.0, 45.0],
            "lon": [-100.0, -80.0],
            "observation_type": ["GS", "ASOS"],
        })
        result = build_neighbor_lists(sites, radius_km=10.0)
        gs1_row = result[result["station_id"] == "GS1"]
        assert gs1_row["neighbors_10km_site_ids"].iloc[0] == ""

    def test_gs_only_keys(self):
        sites = pd.DataFrame({
            "station_id": ["GS1", "A1", "A2"],
            "lat": [40.0, 40.0, 41.0],
            "lon": [-100.0, -100.0, -100.0],
            "observation_type": ["GS", "ASOS", "ASOS"],
        })
        result = build_neighbor_lists(sites, radius_km=100.0)
        assert "GS1" in result["station_id"].values
        # Only GS sites appear as keys
        assert all(
            sid.startswith("GS")
            for sid in result["station_id"].values
        )

    def test_self_exclusion(self):
        sites = pd.DataFrame({
            "station_id": ["GS1"],
            "lat": [40.0],
            "lon": [-100.0],
            "observation_type": ["GS"],
        })
        result = build_neighbor_lists(sites, radius_km=100.0)
        assert result["neighbors_10km_site_ids"].iloc[0] == ""

    def test_asos_as_neighbor_of_gs(self):
        sites = pd.DataFrame({
            "station_id": ["GS1", "A1"],
            "lat": [40.0, 40.001],
            "lon": [-100.0, -100.001],
            "observation_type": ["GS", "ASOS"],
        })
        result = build_neighbor_lists(sites, radius_km=10.0)
        nbrs = result[result["station_id"] == "GS1"]["neighbors_10km_site_ids"].iloc[0]
        assert "A1" in nbrs

    def test_count_column(self):
        sites = pd.DataFrame({
            "station_id": ["GS1", "A1", "A2"],
            "lat": [40.0, 40.001, 40.002],
            "lon": [-100.0, -100.001, -100.002],
            "observation_type": ["GS", "ASOS", "ASOS"],
        })
        result = build_neighbor_lists(sites, radius_km=100.0)
        row = result[result["station_id"] == "GS1"]
        assert row["neighbors_10km_count"].iloc[0] == 2

    def test_sorted_deterministic(self):
        sites = pd.DataFrame({
            "station_id": ["GS1", "B_ASOS", "A_ASOS"],
            "lat": [40.0, 40.001, 40.002],
            "lon": [-100.0, -100.001, -100.002],
            "observation_type": ["GS", "ASOS", "ASOS"],
        })
        result = build_neighbor_lists(sites, radius_km=100.0)
        nbrs = result[result["station_id"] == "GS1"]["neighbors_10km_site_ids"].iloc[0]
        parts = nbrs.split(",")
        assert parts == sorted(parts)
