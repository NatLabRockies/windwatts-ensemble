"""Tests for wem.utils.spatial."""

import math

import numpy as np
import pytest

from wem.utils.spatial import (
    EARTH_RADIUS_KM,
    idw_weights_from_dd,
    pairwise_haversine_km,
    to_webmercator,
    to_xy_lcc,
)


# ---- to_xy_lcc ----

class TestToXyLcc:
    def test_origin_point(self):
        x, y = to_xy_lcc(np.array([-96.0]), np.array([38.47240422490422]))
        assert abs(x[0]) < 1.0  # should be near 0
        assert abs(y[0]) < 1.0

    def test_array_shapes(self):
        lons = np.array([-100.0, -90.0, -80.0])
        lats = np.array([30.0, 40.0, 50.0])
        x, y = to_xy_lcc(lons, lats)
        assert x.shape == (3,)
        assert y.shape == (3,)

    def test_scalar_inputs(self):
        x, y = to_xy_lcc(np.float64(-96.0), np.float64(38.472))
        assert np.isfinite(x)
        assert np.isfinite(y)

    def test_dtype_float64(self):
        x, y = to_xy_lcc(np.array([-96.0]), np.array([38.0]))
        assert x.dtype == np.float64
        assert y.dtype == np.float64


# ---- to_webmercator ----

class TestToWebmercator:
    def test_origin(self):
        x, y = to_webmercator(0.0, 0.0)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6

    def test_known_value(self):
        # London: lon=-0.1278, lat=51.5074
        x, y = to_webmercator(-0.1278, 51.5074)
        assert abs(x - (-14226.5)) < 100  # ~-14226 m
        assert abs(y - 6711568.0) < 1000  # ~6.7M m

    def test_lat_clamp(self):
        _, y1 = to_webmercator(0.0, 89.9)
        _, y2 = to_webmercator(0.0, 85.0511)
        # 89.9 gets clamped to ~85.0511, so y should be close
        assert abs(y1 - y2) < 50

    def test_negative_lat_clamp(self):
        _, y = to_webmercator(0.0, -90.0)
        assert np.isfinite(y)


# ---- idw_weights_from_dd ----

class TestIdwWeights:
    def test_row_sums_to_one(self):
        dd = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        w = idw_weights_from_dd(dd)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)

    def test_equal_distances(self):
        dd = np.array([[10.0, 10.0, 10.0, 10.0]])
        w = idw_weights_from_dd(dd)
        np.testing.assert_allclose(w[0], 0.25, atol=1e-5)

    def test_zero_distance(self):
        dd = np.array([[0.0, 5.0, 10.0, 15.0]])
        w = idw_weights_from_dd(dd)
        assert w[0, 0] == 1.0
        assert np.sum(w[0, 1:]) == 0.0

    def test_closer_gets_more_weight(self):
        dd = np.array([[1.0, 10.0, 100.0, 1000.0]])
        w = idw_weights_from_dd(dd)
        assert w[0, 0] > w[0, 1] > w[0, 2] > w[0, 3]

    def test_shape(self):
        dd = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype="float64")
        w = idw_weights_from_dd(dd)
        assert w.shape == (2, 4)

    def test_dtype_float32(self):
        dd = np.array([[1.0, 2.0, 3.0, 4.0]])
        w = idw_weights_from_dd(dd)
        assert w.dtype == np.float32


# ---- pairwise_haversine_km ----

class TestPairwiseHaversine:
    def test_diagonal_zero(self):
        lat = np.radians(np.array([40.0, 50.0, 60.0]))
        lon = np.radians(np.array([-100.0, -90.0, -80.0]))
        D = pairwise_haversine_km(lat, lon)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_symmetric(self):
        lat = np.radians(np.array([30.0, 45.0]))
        lon = np.radians(np.array([-90.0, -80.0]))
        D = pairwise_haversine_km(lat, lon)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_nyc_la(self):
        # NYC: 40.7128, -74.0060; LA: 34.0522, -118.2437
        lat = np.radians(np.array([40.7128, 34.0522]))
        lon = np.radians(np.array([-74.0060, -118.2437]))
        D = pairwise_haversine_km(lat, lon)
        assert abs(D[0, 1] - 3944) < 50  # ~3944 km

    def test_single_point(self):
        lat = np.radians(np.array([40.0]))
        lon = np.radians(np.array([-100.0]))
        D = pairwise_haversine_km(lat, lon)
        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0

    def test_antipodal(self):
        lat = np.radians(np.array([0.0, 0.0]))
        lon = np.radians(np.array([0.0, 180.0]))
        D = pairwise_haversine_km(lat, lon)
        expected = math.pi * EARTH_RADIUS_KM
        assert abs(D[0, 1] - expected) < 1.0
