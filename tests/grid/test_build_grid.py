"""Tests for wem.grid.build_grid."""

import numpy as np
import pandas as pd
import pytest

from wem.grid.build_grid import assign_indices


class TestAssignIndices:
    def test_grid_id_format(self):
        pts = pd.DataFrame({
            "lat": [40.0, 39.0, 38.0],
            "lon": [-100.0, -99.0, -98.0],
        })
        result = assign_indices(pts)
        # grid_id should be 6 chars
        assert all(len(gid) == 6 for gid in result["grid_id"])

    def test_lat_descending(self):
        pts = pd.DataFrame({
            "lat": [40.0, 39.0, 38.0],
            "lon": [-100.0, -100.0, -100.0],
        })
        result = assign_indices(pts)
        # lat_idx 0 = northernmost (40.0)
        row_40 = result[np.isclose(result["lat"], 40.0)]
        row_38 = result[np.isclose(result["lat"], 38.0)]
        # grid_id first 3 chars = lat_idx; north should have lower idx
        assert row_40["grid_id"].iloc[0][:3] < row_38["grid_id"].iloc[0][:3]

    def test_lon_ascending(self):
        pts = pd.DataFrame({
            "lat": [40.0, 40.0, 40.0],
            "lon": [-100.0, -99.0, -98.0],
        })
        result = assign_indices(pts)
        row_w = result[np.isclose(result["lon"], -100.0)]
        row_e = result[np.isclose(result["lon"], -98.0)]
        assert row_w["grid_id"].iloc[0][3:] < row_e["grid_id"].iloc[0][3:]

    def test_rounding_stability(self):
        pts = pd.DataFrame({
            "lat": [40.0000001, 40.0000002],
            "lon": [-100.0, -100.0],
        })
        result = assign_indices(pts)
        # With default rounding=6, these should map to the same lat_idx
        assert result["grid_id"].iloc[0] == result["grid_id"].iloc[1]

    def test_single_point(self):
        pts = pd.DataFrame({"lat": [35.0], "lon": [-90.0]})
        result = assign_indices(pts)
        assert result["grid_id"].iloc[0] == "000000"

    def test_output_columns(self):
        pts = pd.DataFrame({"lat": [40.0], "lon": [-100.0]})
        result = assign_indices(pts)
        assert set(result.columns) == {"grid_id", "lat", "lon"}

    def test_preserves_original_precision(self):
        pts = pd.DataFrame({"lat": [40.123456789], "lon": [-100.987654321]})
        result = assign_indices(pts)
        assert abs(result["lat"].iloc[0] - 40.123456789) < 1e-9

    def test_multiple_grid_points(self):
        lats = np.arange(40, 43, 0.5)  # 6 lat values
        lons = np.arange(-100, -97, 0.5)  # 6 lon values
        pts = pd.DataFrame({
            "lat": np.repeat(lats, len(lons)),
            "lon": np.tile(lons, len(lats)),
        })
        result = assign_indices(pts)
        assert len(result) == len(lats) * len(lons)
        assert result["grid_id"].nunique() == len(lats) * len(lons)
