"""Tests for wem.experiment.mlp."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wem.experiment.mlp import TabularMLP, run_one_fold_mlp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fold_data(n_per=3, n_features=9):
    """Build minimal long-format fold data with 4 GS + 3 ASOS stations."""
    rng = np.random.default_rng(42)
    sids = (
        ["GS1"] * n_per + ["GS2"] * n_per + ["GS3"] * n_per + ["GS4"] * n_per
        + ["ASOS1"] * n_per + ["ASOS2"] * n_per + ["ASOS3"] * n_per
    )
    n = len(sids)
    station_ids = np.array(sids)
    is_gs = np.array([s.startswith("GS") for s in sids])
    X = rng.normal(5.0, 1.0, (n, n_features)).astype(np.float32)
    y = rng.normal(5.0, 0.5, (n,)).astype(np.float32)
    nbr_map = {"GS1": set(), "GS2": set(), "GS3": set(), "GS4": set()}
    return X, y, station_ids, is_gs, nbr_map


# ===================== TestTabularMLP =====================


class TestTabularMLP:
    def test_forward_shape(self):
        model = TabularMLP(n_features=9, hidden_dims=(128, 64), dropout=0.1)
        x = torch.randn(16, 9)
        out = model(x)
        assert out.shape == (16, 1)

    def test_gradient_flow(self):
        model = TabularMLP(n_features=9, hidden_dims=(128, 64), dropout=0.0)
        x = torch.randn(8, 9, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_custom_hidden_dims(self):
        model = TabularMLP(n_features=5, hidden_dims=(64, 32, 16), dropout=0.1)
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4, 1)


# ===================== TestRunOneFoldMlp =====================


class TestRunOneFoldMlp:
    @pytest.fixture()
    def fold_data(self):
        return _make_fold_data()

    @pytest.mark.slow
    def test_output_shape(self, fold_data):
        X, y, sids, is_gs, nbr_map = fold_data
        sid, test_idx, preds, metrics = run_one_fold_mlp(
            "GS1", X, y, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
            hidden_dims=(16, 8),
        )
        assert sid == "GS1"
        assert preds.shape == (test_idx.size,)

    @pytest.mark.slow
    def test_scalar_output(self, fold_data):
        X, y, sids, is_gs, nbr_map = fold_data
        _, test_idx, preds, _ = run_one_fold_mlp(
            "GS1", X, y, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
            hidden_dims=(16, 8),
        )
        assert preds.ndim == 1
        assert preds.dtype == np.float32

    @pytest.mark.slow
    def test_neighbor_exclusion(self, fold_data):
        X, y, sids, is_gs, nbr_map = fold_data
        nbr_map["GS1"] = {"GS2"}
        sid, test_idx, preds, _ = run_one_fold_mlp(
            "GS1", X, y, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
            hidden_dims=(16, 8),
        )
        assert preds.shape[0] == test_idx.size
        assert test_idx.size > 0

    def test_empty_station_returns_empty(self, fold_data):
        X, y, sids, is_gs, nbr_map = fold_data
        sid, test_idx, preds, metrics = run_one_fold_mlp(
            "NONEXIST", X, y, sids, is_gs, nbr_map,
            seed=42, epochs=5, batch_size=4, patience=3,
            hidden_dims=(16, 8),
        )
        assert test_idx.size == 0
        assert preds.shape[0] == 0
        assert metrics is None
