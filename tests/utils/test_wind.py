"""Tests for wem.utils.wind."""

import numpy as np
import pytest

from wem.utils.wind import gather_unique, uv_from_ws_wd


# ---- uv_from_ws_wd ----

class TestUvFromWsWd:
    @pytest.mark.parametrize("wd,expected_u,expected_v", [
        (0, 10.0, 0.0),    # North: theta=270 -> u=+10, v=0
        (90, 0.0, 10.0),   # East: theta=180 -> u=0, v=+10
        (180, -10.0, 0.0), # South: theta=90 -> u=-10, v=0
        (270, 0.0, -10.0), # West: theta=0 -> u=0, v=-10
    ])
    def test_cardinal_directions(self, wd, expected_u, expected_v):
        ws = np.array([10.0])
        wd_arr = np.array([float(wd)])
        u, v = uv_from_ws_wd(ws, wd_arr)
        assert abs(u[0] - expected_u) < 0.01
        assert abs(v[0] - expected_v) < 0.01

    def test_zero_speed(self):
        u, v = uv_from_ws_wd(np.array([0.0]), np.array([45.0]))
        assert u[0] == 0.0
        assert v[0] == 0.0

    def test_dtype_float32(self):
        u, v = uv_from_ws_wd(np.array([5.0]), np.array([90.0]))
        assert u.dtype == np.float32
        assert v.dtype == np.float32

    def test_shape_preservation(self):
        ws = np.ones((3, 4))
        wd = np.full((3, 4), 180.0)
        u, v = uv_from_ws_wd(ws, wd)
        assert u.shape == (3, 4)
        assert v.shape == (3, 4)

    def test_magnitude_preserved(self):
        ws = np.array([10.0])
        wd = np.array([45.0])
        u, v = uv_from_ws_wd(ws, wd)
        mag = np.sqrt(u[0] ** 2 + v[0] ** 2)
        assert abs(mag - 10.0) < 0.01


# ---- gather_unique ----

class TestGatherUnique:
    def test_basic_dedup(self):
        idx4 = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        uniq, pos_map, cols4 = gather_unique(idx4)
        assert list(uniq) == [0, 1, 2, 3, 4]
        assert len(pos_map) == 5

    def test_no_repeats(self):
        idx4 = np.array([[0, 1, 2, 3]])
        uniq, pos_map, cols4 = gather_unique(idx4)
        assert len(uniq) == 4

    def test_all_same(self):
        idx4 = np.array([[5, 5, 5, 5], [5, 5, 5, 5]])
        uniq, pos_map, cols4 = gather_unique(idx4)
        assert len(uniq) == 1
        assert uniq[0] == 5
        assert np.all(cols4 == 0)

    def test_pos_map_correctness(self):
        idx4 = np.array([[10, 20, 30, 40]])
        uniq, pos_map, cols4 = gather_unique(idx4)
        for g_idx, pos in pos_map.items():
            assert uniq[pos] == g_idx
        # cols4 should remap correctly
        for s in range(idx4.shape[0]):
            for k in range(4):
                assert uniq[cols4[s, k]] == idx4[s, k]
