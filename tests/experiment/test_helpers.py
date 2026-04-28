"""Tests for wem.experiment._helpers — shared experiment pure functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wem.experiment._helpers import (
    build_train_cmd,
    ci95,
    combo_label,
    compute_gs_metrics,
    compute_site_mae,
    compute_sweep_metrics,
    enumerate_subsets,
    features_from_label,
    mae,
    make_pairs_for_feature,
    parse_label_from_path,
    resolve_train_command,
    rmse,
)


# ===================== rmse / mae =====================

class TestRmse:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        y = np.array([0.0, 0.0])
        yhat = np.array([1.0, -1.0])
        assert rmse(y, yhat) == pytest.approx(1.0)


class TestMae:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y = np.array([0.0, 0.0])
        yhat = np.array([3.0, -1.0])
        assert mae(y, yhat) == pytest.approx(2.0)


# ===================== ci95 =====================

class TestCi95:
    def test_too_few(self):
        lo, hi = ci95(np.array([1.0]))
        assert np.isnan(lo) and np.isnan(hi)

    def test_constant_array(self):
        lo, hi = ci95(np.array([5.0, 5.0, 5.0]))
        assert lo == pytest.approx(hi)  # zero width

    def test_known(self):
        x = np.array([10.0, 20.0])
        lo, hi = ci95(x)
        mu = 15.0
        se = float(np.std(x, ddof=1) / np.sqrt(2))
        assert lo == pytest.approx(mu - 1.96 * se)
        assert hi == pytest.approx(mu + 1.96 * se)


# ===================== compute_gs_metrics =====================

class TestComputeGsMetrics:
    def test_missing_file(self, tmp_path):
        r, m, n = compute_gs_metrics(tmp_path / "nope.csv")
        assert r == float("inf")
        assert n == 0

    def test_valid_csv(self, tmp_path):
        df = pd.DataFrame({
            "observation_type": ["GS", "GS", "ASOS"],
            "observation": [1.0, 2.0, 100.0],
            "pred_observation": [1.0, 2.0, 999.0],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        r, m, n = compute_gs_metrics(p)
        assert r == pytest.approx(0.0)
        assert m == pytest.approx(0.0)
        assert n == 2

    def test_missing_columns(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        r, m, n = compute_gs_metrics(p)
        assert r == float("inf")
        assert n == 0

    def test_only_gs_rows_used(self, tmp_path):
        df = pd.DataFrame({
            "observation_type": ["GS", "ASOS"],
            "observation": [10.0, 10.0],
            "pred_observation": [12.0, 999.0],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        r, m, n = compute_gs_metrics(p)
        assert n == 1
        assert m == pytest.approx(2.0)

    def test_empty_gs(self, tmp_path):
        df = pd.DataFrame({
            "observation_type": ["ASOS"],
            "observation": [1.0],
            "pred_observation": [1.0],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        r, m, n = compute_gs_metrics(p)
        assert r == float("inf")
        assert n == 0


# ===================== compute_site_mae =====================

class TestComputeSiteMae:
    def test_missing_file(self, tmp_path):
        df, mean, med, n = compute_site_mae(tmp_path / "nope.csv")
        assert df.empty
        assert n == 0

    def test_per_site(self, tmp_path):
        df = pd.DataFrame({
            "station_id": ["A", "A", "B", "B"],
            "observation_type": ["GS"] * 4,
            "observation": [1.0, 2.0, 3.0, 4.0],
            "pred_observation": [1.5, 2.5, 3.0, 4.0],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        site_df, mean_mae, median_mae, n = compute_site_mae(p)
        assert n == 2
        # Station A MAE = 0.5, Station B MAE = 0.0
        assert mean_mae == pytest.approx(0.25)
        assert median_mae == pytest.approx(0.25)

    def test_missing_columns(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        site_df, mean, med, n = compute_site_mae(p)
        assert site_df.empty

    def test_no_gs_rows(self, tmp_path):
        df = pd.DataFrame({
            "station_id": ["A"],
            "observation_type": ["ASOS"],
            "observation": [1.0],
            "pred_observation": [1.0],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        site_df, mean, med, n = compute_site_mae(p)
        assert n == 0

    def test_mean_vs_median(self, tmp_path):
        # 3 stations with MAEs 0, 0, 6 → mean=2, median=0
        df = pd.DataFrame({
            "station_id": ["A", "B", "C"],
            "observation_type": ["GS"] * 3,
            "observation": [1.0, 1.0, 1.0],
            "pred_observation": [1.0, 1.0, 7.0],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        site_df, mean_mae, median_mae, n = compute_site_mae(p)
        assert n == 3
        assert mean_mae == pytest.approx(2.0)
        assert median_mae == pytest.approx(0.0)


# ===================== compute_sweep_metrics =====================

class TestComputeSweepMetrics:
    def test_pooled(self):
        df = pd.DataFrame({
            "station_id": ["A", "A"],
            "observation_type": ["GS", "GS"],
            "observation": [1.0, 2.0],
            "pred_observation": [1.0, 2.0],
        })
        m = compute_sweep_metrics(df, stationwise=False)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["rows"] == 2

    def test_stationwise(self):
        df = pd.DataFrame({
            "station_id": ["A", "A", "B", "B"],
            "observation_type": ["GS"] * 4,
            "observation": [1.0, 2.0, 10.0, 20.0],
            "pred_observation": [1.0, 2.0, 10.0, 20.0],
        })
        m = compute_sweep_metrics(df, stationwise=True)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["stations"] == 2

    def test_empty(self):
        df = pd.DataFrame({
            "station_id": [],
            "observation_type": [],
            "observation": [],
            "pred_observation": [],
        })
        m = compute_sweep_metrics(df)
        assert np.isnan(m["rmse"])
        assert m["rows"] == 0

    def test_only_gs(self):
        df = pd.DataFrame({
            "station_id": ["A", "B"],
            "observation_type": ["GS", "ASOS"],
            "observation": [1.0, 100.0],
            "pred_observation": [2.0, 999.0],
        })
        m = compute_sweep_metrics(df)
        assert m["rows"] == 1
        assert m["mae"] == pytest.approx(1.0)


# ===================== parse_label_from_path =====================

class TestParseLabelFromPath:
    def test_default_prefix(self):
        p = Path("ml_results_xgb_era5+wtk.csv")
        assert parse_label_from_path(p) == "era5+wtk"

    def test_custom_prefix(self):
        p = Path("ml_results_xgb_aux_latlon+height.csv")
        assert parse_label_from_path(p, prefix="ml_results_xgb_aux_") == "latlon+height"

    def test_no_match(self):
        p = Path("some_other_file.csv")
        assert parse_label_from_path(p) == "some_other_file"

    def test_empty_label(self):
        p = Path("ml_results_xgb_.csv")
        assert parse_label_from_path(p) == "none"


# ===================== features_from_label =====================

class TestFeaturesFromLabel:
    def test_normal(self):
        label, feats = features_from_label("era5+wtk+hrrr")
        assert feats == ("era5", "wtk", "hrrr")

    def test_none(self):
        label, feats = features_from_label("none")
        assert feats == ()

    def test_single(self):
        label, feats = features_from_label("era5")
        assert feats == ("era5",)


# ===================== make_pairs_for_feature =====================

class TestMakePairsForFeature:
    def test_single_pair(self):
        runs = {
            (): {"rmse": 1.0},
            ("era5",): {"rmse": 0.8},
        }
        pairs = make_pairs_for_feature(runs, "era5", "rmse")
        assert len(pairs) == 1
        assert pairs[0][1] == pytest.approx(-0.2)

    def test_multiple_bases(self):
        runs = {
            (): {"rmse": 1.0},
            ("era5",): {"rmse": 0.8},
            ("wtk",): {"rmse": 0.9},
            ("era5", "wtk"): {"rmse": 0.7},
        }
        pairs = make_pairs_for_feature(runs, "era5", "rmse")
        assert len(pairs) == 2  # base () and base (wtk,)

    def test_nan_exclusion(self):
        runs = {
            (): {"rmse": np.nan},
            ("era5",): {"rmse": 0.8},
        }
        pairs = make_pairs_for_feature(runs, "era5", "rmse")
        assert len(pairs) == 0

    def test_feature_already_present(self):
        runs = {
            ("era5",): {"rmse": 0.8},
            ("era5", "wtk"): {"rmse": 0.7},
        }
        # era5 is already in all base sets, no pairs possible
        pairs = make_pairs_for_feature(runs, "era5", "rmse")
        assert len(pairs) == 0


# ===================== combo_label =====================

class TestComboLabel:
    def test_empty(self):
        assert combo_label([]) == "none"

    def test_multiple(self):
        assert combo_label(["a", "b"]) == "a+b"


# ===================== enumerate_subsets =====================

class TestEnumerateSubsets:
    def test_no_empty(self):
        subsets = enumerate_subsets(["a", "b"])
        assert [] not in subsets
        assert len(subsets) == 3  # a, b, a+b

    def test_with_empty(self):
        subsets = enumerate_subsets(["a", "b"], include_empty=True)
        assert [] in subsets
        assert len(subsets) == 4


# ===================== resolve_train_command =====================

class TestResolveTrainCommand:
    def test_default(self):
        cmd = resolve_train_command(None)
        assert cmd == [sys.executable, "-m", "wem.train.loocv_xgb"]

    def test_custom(self):
        cmd = resolve_train_command("python my_script.py")
        assert cmd == ["python", "my_script.py"]


# ===================== build_train_cmd =====================

class TestBuildTrainCmd:
    def test_basic(self, tmp_path):
        cmd = build_train_cmd(
            ["python", "-m", "wem.train.loocv_xgb"],
            tmp_path / "in.csv",
            tmp_path / "out.csv",
        )
        assert "--infile" in cmd
        assert "--val_frac" in cmd
        assert "0.0" in cmd
        assert "--early_stopping_rounds" in cmd
        assert "0" in cmd

    def test_gs_only_flag(self, tmp_path):
        cmd = build_train_cmd(
            ["python", "train.py"],
            tmp_path / "in.csv",
            tmp_path / "out.csv",
            gs_only=True,
        )
        assert "--gs_only" in cmd

    def test_all_params(self, tmp_path):
        cmd = build_train_cmd(
            ["python", "train.py"],
            tmp_path / "in.csv",
            tmp_path / "out.csv",
            params={"learning_rate": 0.05, "max_depth": 10},
            wind_features="hrrr,wtk",
            aux_features="latlon",
            n_jobs_outer=8,
            n_jobs_model=2,
        )
        assert "--learning_rate" in cmd
        assert "--max_depth" in cmd
        assert "--wind_features" in cmd
        assert "--aux_features" in cmd
        assert "--n_jobs_outer" in cmd
