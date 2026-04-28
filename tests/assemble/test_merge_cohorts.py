"""Tests for wem.assemble.merge_cohorts."""

import numpy as np
import pandas as pd
import pytest

from wem.assemble.merge_cohorts import to_long


class TestToLong:
    def _make_wide(self, n_sites=3):
        qcols = [f"q{i:03d}" for i in range(101)]
        rows = []
        for i in range(n_sites):
            row = {
                "station_id": f"S{i:03d}",
                "name": f"Site {i}",
                "lat": 35.0 + i,
                "lon": -100.0 + i,
                "elev_m": 100.0 * i,
                "height_m": 60,
            }
            row.update(dict(zip(qcols, np.linspace(0, 5 + i, 101))))
            rows.append(row)
        return pd.DataFrame(rows)

    def test_shape(self):
        df = self._make_wide(3)
        long, meta = to_long(df, "test")
        assert len(long) == 3 * 101
        assert len(meta) == 3

    def test_qnum_range(self):
        df = self._make_wide(2)
        long, _ = to_long(df, "test")
        assert long["qnum"].min() == 0
        assert long["qnum"].max() == 100

    def test_meta_columns(self):
        df = self._make_wide(1)
        _, meta = to_long(df, "test")
        for col in ["station_id", "height_m", "name", "lat", "lon", "elev_m"]:
            assert col in meta.columns

    def test_flexible_columns(self):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {
            "STATION": ["A"],
            "name": ["Site A"],
            "Latitude": [40.0],
            "Longitude": [-100.0],
            "height_m": [60],
        }
        data.update({c: [0.0] for c in qcols})
        df = pd.DataFrame(data)
        long, meta = to_long(df, "test")
        assert len(long) == 101

    def test_insufficient_qcols_error(self):
        df = pd.DataFrame({
            "station_id": ["A"],
            "lat": [40.0],
            "lon": [-100.0],
            "height_m": [60],
        })
        # Only 10 q columns, not enough
        for i in range(10):
            df[f"q{i:03d}"] = 0.0
        with pytest.raises(ValueError, match="expected q000"):
            to_long(df, "test")

    def test_long_columns(self):
        df = self._make_wide(1)
        long, _ = to_long(df, "test")
        assert set(long.columns) == {"station_id", "height_m", "qnum", "value"}

    def test_numeric_coercion(self):
        df = self._make_wide(1)
        df["height_m"] = df["height_m"].astype(str)  # string "60"
        long, meta = to_long(df, "test")
        assert pd.api.types.is_numeric_dtype(meta["height_m"])

    def test_drops_nan_keys(self):
        df = self._make_wide(2)
        df.loc[1, "lat"] = np.nan
        long, meta = to_long(df, "test")
        assert len(meta) == 1
