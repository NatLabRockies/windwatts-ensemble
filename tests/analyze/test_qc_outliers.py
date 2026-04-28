"""Tests for wem.analyze.qc_outliers: robust_zscores and compute_station_metrics."""

import numpy as np
import pandas as pd
import pytest

from wem.analyze.qc_outliers import compute_station_metrics, robust_zscores


class TestRobustZscores:
    def test_robust_zscores_symmetric(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = robust_zscores(x)
        # The center value (3.0) is the median, so its z-score should be ~0
        np.testing.assert_allclose(z[2], 0.0, atol=1e-10)

    def test_robust_zscores_outlier(self):
        x = np.array([1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 3.0, 100.0])
        z = robust_zscores(x)
        # The value 100.0 is a clear outlier: |z| should be > 3
        assert np.abs(z[-1]) > 3.0

    def test_robust_zscores_constant(self):
        x = np.array([7.0, 7.0, 7.0, 7.0, 7.0])
        z = robust_zscores(x)
        # All-same values have zero spread, so z-scores should be NaN
        assert np.all(np.isnan(z))

    def test_robust_zscores_shape(self):
        x = np.array([10.0, 20.0, 30.0, 40.0])
        z = robust_zscores(x)
        assert z.shape == x.shape


class TestComputeStationMetrics:
    @staticmethod
    def _make_asos_df(n_stations=3, n_quantiles=101):
        """Build a minimal ASOS DataFrame for testing."""
        rows = []
        for i in range(n_stations):
            for q in range(n_quantiles):
                rows.append(
                    {
                        "station_id": f"STN{i:03d}",
                        "observation_type": "ASOS",
                        "qnum": q,
                        "observation": float(q) * 0.1 + i * 0.5,
                        "era5": float(q) * 0.1 + i * 0.5 + 0.01,
                        "lat": 35.0 + i,
                        "lon": -100.0 + i,
                    }
                )
        return pd.DataFrame(rows)

    def test_compute_station_metrics_basic(self):
        df = self._make_asos_df(n_stations=3, n_quantiles=101)
        result = compute_station_metrics(
            df, ref_choice="era5", monotonic_tol=0.0, use_elev_diff=False
        )
        # Should have one row per station
        assert len(result) == 3
        # Must contain these key columns
        for col in ("station_id", "metric", "zscore"):
            assert col in result.columns

    def test_compute_station_metrics_no_asos(self):
        df = pd.DataFrame(
            {
                "station_id": ["GS001"],
                "observation_type": ["GS"],
                "qnum": [0],
                "observation": [5.0],
                "era5": [5.1],
                "lat": [40.0],
                "lon": [-105.0],
            }
        )
        with pytest.raises(ValueError, match="No ASOS rows found"):
            compute_station_metrics(
                df, ref_choice="era5", monotonic_tol=0.0, use_elev_diff=False
            )
