"""Tests for wem.analyze.nn_lookup: to_rad() and nearest_available_heights()."""

import numpy as np
import pytest

from wem.analyze.nn_lookup import nearest_available_heights, to_rad


class TestToRad:
    def test_to_rad_zeros(self):
        result = to_rad(np.array([0]), np.array([0]))
        expected = np.array([[0.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_to_rad_known(self):
        result = to_rad(np.array([90]), np.array([180]))
        expected = np.array([[np.pi / 2, np.pi]])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_to_rad_shape(self):
        n = 7
        lats = np.linspace(-90, 90, n)
        lons = np.linspace(-180, 180, n)
        result = to_rad(lats, lons)
        assert result.shape == (n, 2)


class TestNearestAvailableHeights:
    def test_nearest_heights_exact(self):
        available = np.array([10, 40, 80, 100])
        query = np.array([10, 40, 80, 100])
        result = nearest_available_heights(available, query)
        np.testing.assert_array_equal(result, query)

    def test_nearest_heights_between(self):
        available = np.array([10, 40, 80, 100])
        # 35 is closer to 40 than to 10
        result = nearest_available_heights(available, np.array([35]))
        assert result[0] == 40

    def test_nearest_heights_below(self):
        available = np.array([10, 40, 80, 100])
        # Query below the lowest available should return the lowest
        result = nearest_available_heights(available, np.array([2]))
        assert result[0] == 10

    def test_nearest_heights_above(self):
        available = np.array([10, 40, 80, 100])
        # Query above the highest available should return the highest
        result = nearest_available_heights(available, np.array([999]))
        assert result[0] == 100
