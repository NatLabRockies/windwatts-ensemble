"""Tests for wem.utils.plotting."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import LinearSegmentedColormap

from wem.utils.plotting import (
    make_custom_cmap,
    make_diff_cmap,
    robust_limits,
    symmetric_bias_limit,
    wrap_lon180,
)


# ---- make_custom_cmap ----

class TestMakeCustomCmap:
    def test_returns_colormap(self):
        cmap = make_custom_cmap()
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_correct_name(self):
        cmap = make_custom_cmap()
        assert cmap.name == "wind_0to10plus"

    def test_n_256(self):
        cmap = make_custom_cmap()
        assert cmap.N == 256


# ---- make_diff_cmap ----

class TestMakeDiffCmap:
    def test_returns_colormap(self):
        cmap = make_diff_cmap()
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_correct_name(self):
        cmap = make_diff_cmap()
        assert cmap.name == "diff_rwb_custom"

    def test_n_256(self):
        cmap = make_diff_cmap()
        assert cmap.N == 256


# ---- wrap_lon180 ----

class TestWrapLon180:
    @pytest.mark.parametrize("inp,expected", [
        (0.0, 0.0),
        (360.0, 0.0),
        (270.0, -90.0),
        (-200.0, 160.0),
        (180.0, -180.0),
        (-180.0, -180.0),
        (540.0, -180.0),
    ])
    def test_values(self, inp, expected):
        result = wrap_lon180(np.array([inp]))
        assert abs(result[0] - expected) < 1e-10

    def test_preserves_nan(self):
        result = wrap_lon180(np.array([np.nan, np.inf, 90.0]))
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert abs(result[2] - 90.0) < 1e-10


# ---- robust_limits ----

class TestRobustLimits:
    def test_known_distribution(self):
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(0, 1, 10000))
        lo, hi = robust_limits([s], trim=0.02)
        assert lo < 0
        assert hi > 0
        assert abs(lo) < 3
        assert abs(hi) < 3

    def test_empty(self):
        lo, hi = robust_limits([pd.Series([], dtype=float)])
        assert lo == 0.0
        assert hi == 1.0

    def test_custom_trim(self):
        s = pd.Series(np.linspace(-10, 10, 1000))
        lo, hi = robust_limits([s], trim=0.1)
        assert abs(lo - (-8)) < 0.5
        assert abs(hi - 8) < 0.5


# ---- symmetric_bias_limit ----

class TestSymmetricBiasLimit:
    def test_symmetric(self):
        s = pd.Series(np.linspace(-5, 5, 1000))
        L = symmetric_bias_limit([s], trim=0.02)
        assert L > 0
        assert abs(L - 5.0) < 0.5

    def test_asymmetric(self):
        s = pd.Series(np.linspace(-2, 8, 1000))
        L = symmetric_bias_limit([s], trim=0.02)
        assert L > 0
        assert abs(L - 8.0) < 0.5  # dominated by the positive side

    def test_empty(self):
        L = symmetric_bias_limit([pd.Series([], dtype=float)], trim=0.02)
        assert L == 5.0


# ---- setup_cartopy_axes ----

@pytest.mark.requires_cartopy
class TestSetupCartopyAxes:
    def test_returns_fig_ax(self):
        from wem.utils.plotting import setup_cartopy_axes
        import matplotlib.pyplot as plt
        fig, ax = setup_cartopy_axes(conus=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_conus_extent(self):
        from wem.utils.plotting import setup_cartopy_axes
        import matplotlib.pyplot as plt
        fig, ax = setup_cartopy_axes(conus=True)
        extent = ax.get_extent()
        # Should be roughly [-125, -66.5, 24, 49.5]
        assert extent[0] < -120
        assert extent[1] > -70
        plt.close(fig)


# ---- mask_points_to_us ----

@pytest.mark.requires_cartopy
class TestMaskPointsToUs:
    def test_conus_point(self):
        import wem.utils.plotting as plotting
        plotting._US_PREP = None  # reset cache
        lon = np.array([-100.0])
        lat = np.array([40.0])
        mask = plotting.mask_points_to_us(lon, lat)
        assert mask[0] is True or mask[0] == True

    def test_ocean_point(self):
        import wem.utils.plotting as plotting
        plotting._US_PREP = None
        lon = np.array([0.0])
        lat = np.array([0.0])
        mask = plotting.mask_points_to_us(lon, lat)
        assert mask[0] is False or mask[0] == False
