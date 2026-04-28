"""Tests for wem.experiment.convnet and wide_to_convnet_arrays."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wem.experiment.convnet import CDFConvNet, CDFDataset, run_one_fold_convnet
from wem.experiment.transforms import pivot_to_wide, wide_to_convnet_arrays


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_long_df(n_stations=4, wind_cols=None, obs_type_map=None):
    """Build a synthetic long-format training table for tests."""
    import pandas as pd

    if wind_cols is None:
        wind_cols = ["hrrr", "wtk"]
    if obs_type_map is None:
        obs_type_map = {}

    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_stations):
        sid = f"S{i:03d}"
        obs_type = obs_type_map.get(sid, "GS" if i % 2 == 0 else "ASOS")
        for q in range(101):
            row = {
                "station_id": sid,
                "height_m": 80,
                "qnum": q,
                "lat": 35.0 + i * 0.5,
                "lon": -100.0 + i * 0.5,
                "elevation_m": 300.0 + i * 10,
                "observation_type": obs_type,
                "observation": q * 0.1 + rng.normal(0, 0.01),
                "neighbors_10km_site_ids": "",
            }
            for wc in wind_cols:
                row[wc] = q * 0.1 + rng.normal(0, 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_convnet_arrays(n_rows=20, n_channels=3, n_aux=4):
    """Build synthetic arrays matching convnet input shapes."""
    rng = np.random.default_rng(123)
    cdf = np.sort(rng.uniform(0, 15, (n_rows, n_channels, 101)), axis=2).astype(np.float32)
    aux = rng.normal(0, 1, (n_rows, n_aux)).astype(np.float32)
    targets = np.sort(rng.uniform(0, 15, (n_rows, 101)), axis=1).astype(np.float32)
    return cdf, aux, targets


# ===================== TestCDFDataset =====================


class TestCDFDataset:
    def test_len(self):
        cdf, aux, tgt = _make_convnet_arrays(n_rows=10)
        ds = CDFDataset(cdf, aux, tgt)
        assert len(ds) == 10

    def test_shapes(self):
        cdf, aux, tgt = _make_convnet_arrays(n_rows=10, n_channels=3, n_aux=4)
        ds = CDFDataset(cdf, aux, tgt)
        c, a, t = ds[0]
        assert c.shape == (3, 101)
        assert a.shape == (4,)
        assert t.shape == (101,)

    def test_dtypes(self):
        cdf, aux, tgt = _make_convnet_arrays(n_rows=5)
        ds = CDFDataset(cdf, aux, tgt)
        c, a, t = ds[0]
        assert c.dtype == torch.float32
        assert a.dtype == torch.float32
        assert t.dtype == torch.float32


# ===================== TestCDFConvNet =====================


class TestCDFConvNet:
    def test_forward_shape(self):
        model = CDFConvNet(n_aux_features=4, dropout=0.1, n_conv_layers=3)
        cdf = torch.randn(8, 3, 101)
        aux = torch.randn(8, 4)
        out = model(cdf, aux)
        assert out.shape == (8, 101)

    def test_monotonicity_guarantee(self):
        model = CDFConvNet(n_aux_features=4, dropout=0.0, n_conv_layers=3)
        cdf = torch.randn(16, 3, 101)
        aux = torch.randn(16, 4)
        with torch.no_grad():
            out = model(cdf, aux)
        diffs = out[:, 1:] - out[:, :-1]
        assert torch.all(diffs >= 0), "Output must be monotonically non-decreasing"

    def test_non_negativity(self):
        model = CDFConvNet(n_aux_features=4, dropout=0.0, n_conv_layers=3)
        cdf = torch.randn(16, 3, 101)
        aux = torch.randn(16, 4)
        with torch.no_grad():
            out = model(cdf, aux)
        assert torch.all(out >= 0), "Output must be non-negative"

    def test_gradient_flow(self):
        model = CDFConvNet(n_aux_features=4, dropout=0.0, n_conv_layers=3)
        cdf = torch.randn(4, 3, 101, requires_grad=True)
        aux = torch.randn(4, 4, requires_grad=True)
        out = model(cdf, aux)
        loss = out.sum()
        loss.backward()
        assert cdf.grad is not None
        assert aux.grad is not None
        assert torch.any(cdf.grad != 0)

    def test_n_conv_layers_2(self):
        model = CDFConvNet(n_aux_features=4, dropout=0.1, n_conv_layers=2)
        cdf = torch.randn(4, 3, 101)
        aux = torch.randn(4, 4)
        out = model(cdf, aux)
        assert out.shape == (4, 101)

    def test_no_aux_features(self):
        model = CDFConvNet(n_aux_features=0, dropout=0.1, n_conv_layers=3)
        cdf = torch.randn(4, 3, 101)
        aux = torch.randn(4, 0)
        out = model(cdf, aux)
        assert out.shape == (4, 101)


# ===================== TestRunOneFoldConvnet =====================


class TestRunOneFoldConvnet:
    @pytest.fixture()
    def fold_data(self):
        """Build minimal data with 4 GS + 3 ASOS stations."""
        rng = np.random.default_rng(42)
        n_per = 3
        sids = (
            ["GS1"] * n_per + ["GS2"] * n_per + ["GS3"] * n_per + ["GS4"] * n_per
            + ["ASOS1"] * n_per + ["ASOS2"] * n_per + ["ASOS3"] * n_per
        )
        n = len(sids)
        station_ids = np.array(sids)
        is_gs = np.array([s.startswith("GS") for s in sids])
        cdf = np.sort(rng.uniform(0, 15, (n, 2, 101)), axis=2).astype(np.float32)
        aux = rng.normal(0, 1, (n, 4)).astype(np.float32)
        targets = np.sort(rng.uniform(0, 15, (n, 101)), axis=1).astype(np.float32)
        nbr_map = {"GS1": set(), "GS2": set(), "GS3": set(), "GS4": set()}
        return cdf, aux, targets, station_ids, is_gs, nbr_map

    @pytest.mark.slow
    def test_output_shape(self, fold_data):
        cdf, aux, tgt, sids, is_gs, nbr_map = fold_data
        sid, test_idx, preds, metrics = run_one_fold_convnet(
            "GS1", cdf, aux, tgt, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
        )
        assert sid == "GS1"
        assert preds.shape == (test_idx.size, 101)

    @pytest.mark.slow
    def test_monotonicity(self, fold_data):
        cdf, aux, tgt, sids, is_gs, nbr_map = fold_data
        _, _, preds, _ = run_one_fold_convnet(
            "GS1", cdf, aux, tgt, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
        )
        for row in preds:
            diffs = np.diff(row)
            assert np.all(diffs >= -1e-6), "Monotonicity violated"

    @pytest.mark.slow
    def test_neighbor_exclusion(self, fold_data):
        cdf, aux, tgt, sids, is_gs, nbr_map = fold_data
        nbr_map["GS1"] = {"GS2"}
        sid, test_idx, preds, _ = run_one_fold_convnet(
            "GS1", cdf, aux, tgt, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
        )
        assert preds.shape[0] == test_idx.size
        assert test_idx.size > 0

    def test_empty_station_returns_empty(self, fold_data):
        cdf, aux, tgt, sids, is_gs, nbr_map = fold_data
        sid, test_idx, preds, metrics = run_one_fold_convnet(
            "NONEXIST", cdf, aux, tgt, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
        )
        assert test_idx.size == 0
        assert preds.shape[0] == 0
        assert metrics is None


# ===================== TestWideToConvnetArrays =====================


class TestWideToConvnetArrays:
    def test_shapes_2_channels(self):
        df = _make_long_df(n_stations=4, wind_cols=["hrrr", "wtk"])
        wide = pivot_to_wide(df, ["hrrr", "wtk"])
        cdf, aux, tgt = wide_to_convnet_arrays(wide, ["hrrr", "wtk"])
        assert cdf.shape == (len(wide), 2, 101)
        assert aux.shape == (len(wide), 4)  # lat, lon, height_m, elevation_m
        assert tgt.shape == (len(wide), 101)

    def test_shapes_with_gwa(self):
        df = _make_long_df(n_stations=4, wind_cols=["hrrr", "wtk"])
        df["gwa_interp"] = 5.0
        wide = pivot_to_wide(df, ["hrrr", "wtk"])
        cdf, aux, tgt = wide_to_convnet_arrays(
            wide, ["hrrr", "wtk"], include_gwa=True, gwa_feature_name="gwa_interp",
        )
        assert aux.shape[1] == 5  # lat, lon, height_m, elevation_m, gwa_interp

    def test_value_correctness(self):
        df = _make_long_df(n_stations=3, wind_cols=["hrrr", "wtk"])
        wide = pivot_to_wide(df, ["hrrr", "wtk"])
        cdf, aux, tgt = wide_to_convnet_arrays(wide, ["hrrr", "wtk"])

        # Check that CDF channel 0 (hrrr) matches hrrr_q000..hrrr_q100
        hrrr_cols = [f"hrrr_q{q:03d}" for q in range(101)]
        expected = wide[hrrr_cols].to_numpy(dtype=np.float32)
        np.testing.assert_array_almost_equal(cdf[:, 0, :], expected)

        # Check targets match obs_q000..obs_q100
        obs_cols = [f"obs_q{q:03d}" for q in range(101)]
        expected_tgt = wide[obs_cols].to_numpy(dtype=np.float32)
        np.testing.assert_array_almost_equal(tgt, expected_tgt)
