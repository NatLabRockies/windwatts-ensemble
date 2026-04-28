"""Tests for wem.utils.raster."""

import numpy as np
import pytest

pytestmark = pytest.mark.requires_rasterio


class TestSampleRasterPoints:
    def test_none_path_returns_nan(self):
        from wem.utils.raster import sample_raster_points
        lons = np.array([1.0, 2.0, 3.0])
        lats = np.array([4.0, 5.0, 6.0])
        result = sample_raster_points(None, lons, lats)
        assert result.dtype == np.float32
        assert np.all(np.isnan(result))
        assert result.shape == (3,)

    def test_basic_sampling(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        from wem.utils.raster import sample_raster_points

        # Create a simple 10x10 raster
        p = tmp_path / "test.tif"
        data = np.arange(100, dtype="float32").reshape(10, 10)
        transform = from_bounds(-110, 30, -100, 40, 10, 10)
        with rasterio.open(
            p, "w", driver="GTiff", height=10, width=10,
            count=1, dtype="float32", crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        lons = np.array([-105.0])
        lats = np.array([35.0])
        result = sample_raster_points(p, lons, lats)
        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_nodata_to_nan(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        from wem.utils.raster import sample_raster_points

        p = tmp_path / "nodata.tif"
        data = np.full((10, 10), -9999.0, dtype="float32")
        transform = from_bounds(-110, 30, -100, 40, 10, 10)
        with rasterio.open(
            p, "w", driver="GTiff", height=10, width=10,
            count=1, dtype="float32", crs="EPSG:4326",
            transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        lons = np.array([-105.0])
        lats = np.array([35.0])
        result = sample_raster_points(p, lons, lats)
        assert np.isnan(result[0])

    def test_large_negative_to_nan(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        from wem.utils.raster import sample_raster_points

        p = tmp_path / "big_neg.tif"
        data = np.full((10, 10), -1e21, dtype="float32")
        transform = from_bounds(-110, 30, -100, 40, 10, 10)
        with rasterio.open(
            p, "w", driver="GTiff", height=10, width=10,
            count=1, dtype="float32", crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        lons = np.array([-105.0])
        lats = np.array([35.0])
        result = sample_raster_points(p, lons, lats)
        assert np.isnan(result[0])

    def test_dtype_float32(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        from wem.utils.raster import sample_raster_points

        p = tmp_path / "dtype.tif"
        data = np.ones((5, 5), dtype="float32") * 42.0
        transform = from_bounds(-110, 30, -100, 40, 5, 5)
        with rasterio.open(
            p, "w", driver="GTiff", height=5, width=5,
            count=1, dtype="float32", crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        result = sample_raster_points(p, np.array([-105.0]), np.array([35.0]))
        assert result.dtype == np.float32
