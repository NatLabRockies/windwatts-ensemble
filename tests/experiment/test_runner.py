"""Tests for wem.experiment.runner."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

try:
    import xgboost  # noqa: F401
except Exception as exc:
    pytest.skip(f"xgboost unavailable: {exc}", allow_module_level=True)

from wem.experiment.runner import (
    EXPERIMENTS,
    ExperimentType,
    build_hybrid_tail_features,
    build_long_features,
    build_wide_features,
    run_one_fold_hybrid,
    run_one_fold_wide,
)
from wem.experiment.transforms import (
    enrich_with_cdf,
    enrich_with_cdf_subset,
    pivot_to_wide,
)
from wem.utils.ml import balance_indices, fold_seed


# ---------------------------------------------------------------------------
# Helpers for building synthetic data
# ---------------------------------------------------------------------------


def _make_long_df(
    n_stations: int = 4,
    n_heights: int = 2,
    wind_cols: list[str] | None = None,
    obs_type_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic long-format training table."""
    if wind_cols is None:
        wind_cols = ["hrrr", "wtk"]
    if obs_type_map is None:
        obs_type_map = {}

    rng = np.random.default_rng(42)
    rows = []
    heights = [40, 80][:n_heights]
    for i in range(n_stations):
        sid = f"S{i:03d}"
        obs_type = obs_type_map.get(sid, "GS" if i % 2 == 0 else "ASOS")
        for h in heights:
            for q in range(101):
                row = {
                    "station_id": sid,
                    "height_m": h,
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


def _make_wide_arrays(n_rows: int = 20, n_features: int = 10):
    """Build synthetic wide-format arrays for fold tests."""
    rng = np.random.default_rng(123)
    X = rng.normal(5.0, 1.0, (n_rows, n_features)).astype(np.float32)
    Y = rng.normal(5.0, 0.5, (n_rows, 101)).astype(np.float32)
    # Make Y monotonically increasing across quantiles (valid CDFs)
    Y = np.sort(Y, axis=1)
    return X, Y


# ===================== TestExperimentRegistry =====================


class TestExperimentRegistry:
    def test_all_types_present(self):
        assert "baseline" in EXPERIMENTS
        assert "enriched" in EXPERIMENTS
        assert "wide" in EXPERIMENTS
        assert "convnet" in EXPERIMENTS
        assert "mlp" in EXPERIMENTS
        assert "hybrid" in EXPERIMENTS

    def test_formats(self):
        assert EXPERIMENTS["baseline"].format == "long"
        assert EXPERIMENTS["enriched"].format == "long"
        assert EXPERIMENTS["wide"].format == "wide"
        assert EXPERIMENTS["convnet"].format == "convnet"
        assert EXPERIMENTS["mlp"].format == "mlp"
        assert EXPERIMENTS["hybrid"].format == "hybrid"


# ===================== TestBuildLongFeatures =====================


class TestBuildLongFeatures:
    def test_baseline_features(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        feat_cols, mon_str, rfm = build_long_features(
            df, ["hrrr", "wtk", "wtk_led_conus"], False, None, enriched=False,
        )
        assert feat_cols == ["qnum", "hrrr", "wtk", "wtk_led_conus",
                             "lat", "lon", "height_m", "elevation_m"]
        assert mon_str == "(1,0,0,0,0,0,0,0)"
        assert rfm.all()  # All features require finite

    def test_enriched_features(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        df = enrich_with_cdf(df, ["hrrr", "wtk", "wtk_led_conus"])
        feat_cols, mon_str, rfm = build_long_features(
            df, ["hrrr", "wtk", "wtk_led_conus"], False, None, enriched=True,
        )
        # qnum(1) + 3*101 CDF(303) + 4 aux(4) = 308
        assert len(feat_cols) == 308
        assert feat_cols[0] == "qnum"
        assert "hrrr_q050" in feat_cols
        assert "wtk_q050" in feat_cols
        assert "wtk_led_conus_q050" in feat_cols
        # Raw wind columns should NOT be features
        assert "hrrr" not in feat_cols
        # Monotonic: +1 on qnum (index 0), 0 on everything else
        assert mon_str.startswith("(1,")
        parts = mon_str.strip("()").split(",")
        assert parts[0] == "1"
        assert all(p == "0" for p in parts[1:])

    def test_with_gwa(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr"])
        df["gwa_interp"] = 5.0
        feat_cols, mon_str, rfm = build_long_features(
            df, ["hrrr"], True, "gwa_interp", enriched=False,
        )
        assert "gwa_interp" in feat_cols
        # GWA should NOT require finite
        gwa_idx = feat_cols.index("gwa_interp")
        assert not rfm[gwa_idx]
        # Other features should require finite
        assert rfm[0]  # qnum

    def test_baseline_with_gwa_count(self):
        """Baseline + GWA should have 9 features total."""
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        df["gwa_interp"] = 5.0
        feat_cols, _, _ = build_long_features(
            df, ["hrrr", "wtk", "wtk_led_conus"], True, "gwa_interp", enriched=False,
        )
        assert len(feat_cols) == 9

    def test_enriched_with_gwa_count(self):
        """Enriched + GWA should have 309 features total."""
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        df["gwa_interp"] = 5.0
        df = enrich_with_cdf(df, ["hrrr", "wtk", "wtk_led_conus"])
        feat_cols, _, _ = build_long_features(
            df, ["hrrr", "wtk", "wtk_led_conus"], True, "gwa_interp", enriched=True,
        )
        assert len(feat_cols) == 309


# ===================== TestBuildWideFeatures =====================


class TestBuildWideFeatures:
    def test_feature_count(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        wide = pivot_to_wide(df, ["hrrr", "wtk", "wtk_led_conus"])
        feat_cols, obs_cols = build_wide_features(
            wide, ["hrrr", "wtk", "wtk_led_conus"], False, None,
        )
        # 3*101 CDF + 4 aux = 307
        assert len(feat_cols) == 307
        assert len(obs_cols) == 101

    def test_no_qnum(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])
        feat_cols, _ = build_wide_features(wide, ["hrrr"], False, None)
        assert "qnum" not in feat_cols

    def test_with_gwa(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        df["gwa_interp"] = 5.0
        wide = pivot_to_wide(df, ["hrrr", "wtk", "wtk_led_conus"])
        feat_cols, _ = build_wide_features(
            wide, ["hrrr", "wtk", "wtk_led_conus"], True, "gwa_interp",
        )
        assert "gwa_interp" in feat_cols
        # 3*101 + 4 aux + 1 GWA = 308
        assert len(feat_cols) == 308


# ===================== TestRunOneFoldWide =====================


class TestRunOneFoldWide:
    @pytest.fixture()
    def fold_data(self):
        """Build minimal wide-format fold data with 4 GS + 3 ASOS stations."""
        n = 20
        X, Y = _make_wide_arrays(n_rows=n, n_features=10)
        station_ids = np.array(
            ["GS1"] * 3 + ["GS2"] * 3 + ["GS3"] * 3 + ["GS4"] * 2
            + ["ASOS1"] * 3 + ["ASOS2"] * 3 + ["ASOS3"] * 3
        )
        is_gs = np.array([s.startswith("GS") for s in station_ids])
        nbr_map = {"GS1": set(), "GS2": set(), "GS3": set(), "GS4": set()}
        params = {
            "learning_rate": 0.3,
            "max_depth": 3,
            "n_estimators": 5,
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": 42,
        }
        return X, Y, station_ids, is_gs, nbr_map, params

    def test_output_shape(self, fold_data):
        X, Y, sids, is_gs, nbr_map, params = fold_data
        sid, test_idx, preds, metrics = run_one_fold_wide(
            "GS1", X, Y, sids, is_gs, nbr_map, params, seed=42,
        )
        assert sid == "GS1"
        assert preds.shape == (test_idx.size, 101)

    def test_monotonicity_enforced(self, fold_data):
        X, Y, sids, is_gs, nbr_map, params = fold_data
        _, _, preds, _ = run_one_fold_wide(
            "GS1", X, Y, sids, is_gs, nbr_map, params, seed=42,
        )
        for row in preds:
            diffs = np.diff(row)
            assert np.all(diffs >= -1e-7), "Monotonicity violated"

    def test_exclusion_of_test_station(self, fold_data):
        X, Y, sids, is_gs, nbr_map, params = fold_data
        nbr_map["GS1"] = {"GS2"}
        sid, test_idx, preds, _ = run_one_fold_wide(
            "GS1", X, Y, sids, is_gs, nbr_map, params, seed=42,
        )
        assert preds.shape[0] == test_idx.size
        assert test_idx.size > 0

    def test_empty_station_returns_empty(self, fold_data):
        X, Y, sids, is_gs, nbr_map, params = fold_data
        sid, test_idx, preds, metrics = run_one_fold_wide(
            "NONEXIST", X, Y, sids, is_gs, nbr_map, params, seed=42,
        )
        assert test_idx.size == 0
        assert preds.shape[0] == 0
        assert metrics is None

    def test_metrics_returned(self, fold_data):
        X, Y, sids, is_gs, nbr_map, params = fold_data
        _, _, _, metrics = run_one_fold_wide(
            "GS1", X, Y, sids, is_gs, nbr_map, params, seed=42,
        )
        assert metrics is not None
        rmse, mae = metrics
        assert rmse >= 0
        assert mae >= 0


# ===================== TestBuildHybridTailFeatures =====================


class TestBuildHybridTailFeatures:
    def test_feature_count_with_gwa(self):
        """3 wind cols * (1 raw + 2 CDF) + qnum + 4 aux + GWA = 15."""
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        df["gwa_interp"] = 5.0
        df = enrich_with_cdf_subset(df, ["hrrr", "wtk", "wtk_led_conus"])
        feat_cols, rfm = build_hybrid_tail_features(
            df, ["hrrr", "wtk", "wtk_led_conus"], True, "gwa_interp",
        )
        assert len(feat_cols) == 15

    def test_cdf_context_present(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        df = enrich_with_cdf_subset(df, ["hrrr", "wtk", "wtk_led_conus"])
        feat_cols, _ = build_hybrid_tail_features(
            df, ["hrrr", "wtk", "wtk_led_conus"], False, None,
        )
        assert "hrrr_q050" in feat_cols
        assert "hrrr_q090" in feat_cols
        assert "wtk_q050" in feat_cols
        assert "wtk_led_conus_q090" in feat_cols

    def test_gwa_not_required_finite(self):
        df = _make_long_df(n_stations=3, n_heights=1, wind_cols=["hrrr"])
        df["gwa_interp"] = 5.0
        df = enrich_with_cdf_subset(df, ["hrrr"])
        feat_cols, rfm = build_hybrid_tail_features(
            df, ["hrrr"], True, "gwa_interp",
        )
        gwa_idx = feat_cols.index("gwa_interp")
        assert not rfm[gwa_idx]


# ===================== TestRunOneFoldHybrid =====================


class TestRunOneFoldHybrid:
    @pytest.fixture()
    def hybrid_data(self):
        """Build minimal long-format data for hybrid fold tests."""
        wind_cols = ["hrrr", "wtk"]
        df = _make_long_df(n_stations=6, n_heights=1, wind_cols=wind_cols,
                           obs_type_map={f"S{i:03d}": "GS" if i % 2 == 0 else "ASOS" for i in range(6)})
        df = enrich_with_cdf_subset(df, wind_cols)

        base_feat = ["qnum"] + wind_cols + ["lat", "lon", "height_m", "elevation_m"]
        tail_feat = ["qnum"] + wind_cols + ["hrrr_q050", "hrrr_q090", "wtk_q050", "wtk_q090",
                     "lat", "lon", "height_m", "elevation_m"]

        X_base = df[base_feat].to_numpy(dtype="float32")
        X_tail = df[tail_feat].to_numpy(dtype="float32")
        y_full = df["observation"].to_numpy(dtype="float32")
        station_ids = df["station_id"].astype(str).to_numpy()
        is_gs = np.array([s in {"S000", "S002", "S004"} for s in station_ids])
        qnums = df["qnum"].to_numpy(dtype=int)
        nbr_map = {s: set() for s in ["S000", "S002", "S004"]}

        base_xgb = {
            "n_estimators": 5, "learning_rate": 0.3, "max_depth": 3,
            "objective": "reg:absoluteerror", "tree_method": "hist",
            "n_jobs": 1, "random_state": 42, "early_stopping_rounds": 0,
            "monotone_constraints": "(1," + ",".join(["0"] * (len(base_feat) - 1)) + ")",
        }
        base_args = {"balance_strategy": "downsample", "val_frac": 0.0,
                     "seed": 42, "xgb_params": base_xgb}

        tail_xgb = {
            "n_estimators": 5, "learning_rate": 0.3, "max_depth": 3,
            "objective": "reg:absoluteerror", "tree_method": "hist",
            "n_jobs": 1, "random_state": 42,
        }

        rf_base = np.ones(len(base_feat), dtype=bool)
        rf_tail = np.ones(len(tail_feat), dtype=bool)

        return (X_base, X_tail, y_full, qnums, station_ids, is_gs,
                nbr_map, base_args, tail_xgb, rf_base, rf_tail)

    def test_output_shape(self, hybrid_data):
        (X_base, X_tail, y_full, qnums, sids, is_gs,
         nbr_map, base_args, tail_xgb, rf_base, rf_tail) = hybrid_data
        sid, test_idx, pred, metrics = run_one_fold_hybrid(
            "S000", X_base, X_tail, y_full, qnums, sids, is_gs,
            nbr_map, base_args, tail_xgb, rf_base, rf_tail,
            tail_cutoff=95, log_floor=1e-6,
        )
        assert sid == "S000"
        assert pred.shape == test_idx.shape
        assert test_idx.size == 101  # one height, 101 quantiles

    def test_tail_predictions_positive(self, hybrid_data):
        """exp() guarantees positive tail predictions."""
        (X_base, X_tail, y_full, qnums, sids, is_gs,
         nbr_map, base_args, tail_xgb, rf_base, rf_tail) = hybrid_data
        _, test_idx, pred, _ = run_one_fold_hybrid(
            "S000", X_base, X_tail, y_full, qnums, sids, is_gs,
            nbr_map, base_args, tail_xgb, rf_base, rf_tail,
            tail_cutoff=95, log_floor=1e-6,
        )
        tail_mask = qnums[test_idx] >= 95
        tail_preds = pred[tail_mask]
        assert np.all(tail_preds > 0)

    def test_base_range_matches_standalone(self, hybrid_data):
        """Base-range predictions should be identical to standalone baseline."""
        from wem.train.loocv_xgb import run_one_fold

        (X_base, X_tail, y_full, qnums, sids, is_gs,
         nbr_map, base_args, tail_xgb, rf_base, rf_tail) = hybrid_data

        # Standalone baseline
        _, base_test_idx, base_pred, _ = run_one_fold(
            "S000", X_base, y_full, sids, is_gs, nbr_map,
            base_args, require_finite_mask=rf_base,
        )

        # Hybrid
        _, hybrid_test_idx, hybrid_pred, _ = run_one_fold_hybrid(
            "S000", X_base, X_tail, y_full, qnums, sids, is_gs,
            nbr_map, base_args, tail_xgb, rf_base, rf_tail,
            tail_cutoff=95, log_floor=1e-6,
        )

        # Base range (q < 95) should be identical
        base_mask = qnums[hybrid_test_idx] < 95
        np.testing.assert_array_equal(hybrid_pred[base_mask], base_pred[base_mask])

    def test_empty_station_returns_empty(self, hybrid_data):
        (X_base, X_tail, y_full, qnums, sids, is_gs,
         nbr_map, base_args, tail_xgb, rf_base, rf_tail) = hybrid_data
        sid, test_idx, pred, metrics = run_one_fold_hybrid(
            "NONEXIST", X_base, X_tail, y_full, qnums, sids, is_gs,
            nbr_map, base_args, tail_xgb, rf_base, rf_tail,
            tail_cutoff=95, log_floor=1e-6,
        )
        assert test_idx.size == 0
        assert metrics is None


# ===================== TestMainE2E =====================


class TestMainE2E:
    def _make_input(self, tmp_path, n_stations=6, wind_cols=None):
        """Write synthetic input CSV and return (infile, outfile) paths."""
        if wind_cols is None:
            wind_cols = ["hrrr", "wtk"]
        obs_map = {}
        for i in range(n_stations):
            sid = f"S{i:03d}"
            obs_map[sid] = "GS" if i % 2 == 0 else "ASOS"
        df = _make_long_df(
            n_stations=n_stations, n_heights=1, wind_cols=wind_cols,
            obs_type_map=obs_map,
        )
        infile = tmp_path / "input.csv"
        df.to_csv(infile, index=False)
        outfile = tmp_path / "output.csv"
        return infile, outfile

    def test_baseline_e2e(self, tmp_path):
        infile, outfile = self._make_input(tmp_path)
        with patch("sys.argv", [
            "prog", "baseline",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr,wtk",
            "--n-jobs", "1",
            "--n-estimators", "5",
            "--max-depth", "3",
            "--learning-rate", "0.3",
            "--seed", "42",
        ]):
            from wem.experiment.runner import main
            main()

        assert outfile.exists()
        result = pd.read_csv(outfile)
        assert "pred_observation" in result.columns
        assert "qnum" in result.columns
        assert "station_id" in result.columns

    def test_enriched_e2e(self, tmp_path):
        infile, outfile = self._make_input(tmp_path)
        with patch("sys.argv", [
            "prog", "enriched",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr,wtk",
            "--n-jobs", "1",
            "--n-estimators", "5",
            "--max-depth", "3",
            "--learning-rate", "0.3",
            "--seed", "42",
        ]):
            from wem.experiment.runner import main
            main()

        assert outfile.exists()
        result = pd.read_csv(outfile)
        assert "pred_observation" in result.columns
        # CDF columns should be stripped from output
        assert "hrrr_q050" not in result.columns
        assert "wtk_q050" not in result.columns

    def test_wide_e2e(self, tmp_path):
        infile, outfile = self._make_input(tmp_path)
        with patch("sys.argv", [
            "prog", "wide",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr,wtk",
            "--n-jobs", "1",
            "--n-estimators", "5",
            "--max-depth", "3",
            "--learning-rate", "0.3",
            "--seed", "42",
        ]):
            from wem.experiment.runner import main
            main()

        assert outfile.exists()
        result = pd.read_csv(outfile)
        assert "pred_observation" in result.columns
        assert "qnum" in result.columns
        assert "station_id" in result.columns

    def test_convnet_e2e(self, tmp_path):
        pytest.importorskip("torch")
        infile, outfile = self._make_input(tmp_path)
        with patch("sys.argv", [
            "prog", "convnet",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr,wtk",
            "--epochs", "5",
            "--batch-size", "4",
            "--patience", "3",
            "--seed", "42",
        ]):
            from wem.experiment.runner import main
            main()

        assert outfile.exists()
        result = pd.read_csv(outfile)
        assert "pred_observation" in result.columns
        assert "qnum" in result.columns
        assert "station_id" in result.columns

    def test_mlp_e2e(self, tmp_path):
        pytest.importorskip("torch")
        infile, outfile = self._make_input(tmp_path)
        with patch("sys.argv", [
            "prog", "mlp",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr,wtk",
            "--epochs", "5",
            "--batch-size", "4",
            "--patience", "3",
            "--hidden-dims", "16", "8",
            "--seed", "42",
        ]):
            from wem.experiment.runner import main
            main()

        assert outfile.exists()
        result = pd.read_csv(outfile)
        assert "pred_observation" in result.columns
        assert "qnum" in result.columns
        assert "station_id" in result.columns

    def test_overwrite_blocked(self, tmp_path):
        infile, outfile = self._make_input(tmp_path)
        outfile.write_text("dummy")

        with patch("sys.argv", [
            "prog", "baseline",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr",
            "--n-jobs", "1",
        ]):
            from wem.experiment.runner import main
            with pytest.raises(SystemExit, match="use --overwrite"):
                main()

    def test_overwrite_allowed(self, tmp_path):
        infile, outfile = self._make_input(tmp_path)
        outfile.write_text("dummy")

        with patch("sys.argv", [
            "prog", "baseline",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr,wtk",
            "--n-jobs", "1",
            "--n-estimators", "5",
            "--max-depth", "3",
            "--learning-rate", "0.3",
            "--overwrite",
        ]):
            from wem.experiment.runner import main
            main()

        assert outfile.exists()
        result = pd.read_csv(outfile)
        assert "pred_observation" in result.columns

    def test_hybrid_e2e(self, tmp_path):
        infile, outfile = self._make_input(tmp_path)
        with patch("sys.argv", [
            "prog", "hybrid",
            "--infile", str(infile),
            "--outfile", str(outfile),
            "--wind-features", "hrrr,wtk",
            "--n-jobs", "1",
            "--n-estimators", "5",
            "--max-depth", "3",
            "--learning-rate", "0.3",
            "--seed", "42",
            "--tail-cutoff", "95",
        ]):
            from wem.experiment.runner import main
            main()

        assert outfile.exists()
        result = pd.read_csv(outfile)
        assert "pred_observation" in result.columns
        assert "qnum" in result.columns
        assert "station_id" in result.columns
        # CDF subset columns should be stripped from output
        assert "hrrr_q050" not in result.columns
        assert "wtk_q090" not in result.columns
