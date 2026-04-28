"""Tests for wem.experiment.analyze_sweep — sweep analysis and plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wem.experiment.analyze_sweep import (
    AUX_PLOT_CONFIG,
    WIND_PLOT_CONFIG,
    analyze_sweep_results,
    plot_combo_presence_heatmap,
    plot_marginal_bar_means,
    plot_marginal_box,
    plot_metric_by_count,
    plot_overall_bar,
    savefig,
)


# ===================== config validation =====================

class TestPlotConfigs:
    def test_wind_config_has_keys(self):
        for key in ("feature_labels", "box_order", "box_palette",
                     "heatmap_mode", "file_prefix"):
            assert key in WIND_PLOT_CONFIG

    def test_aux_config_has_keys(self):
        for key in ("feature_labels", "box_order", "box_palette",
                     "heatmap_mode", "file_prefix"):
            assert key in AUX_PLOT_CONFIG


# ===================== savefig =====================

class TestSavefig:
    def test_creates_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "test.png"
        savefig(fig, out, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 0


# ===================== plot functions =====================

class TestPlotOverallBar:
    def test_creates_png(self, tmp_path):
        df = pd.DataFrame({"label": ["a", "b"], "rmse": [0.5, 0.6]})
        out = tmp_path / "bar.png"
        plot_overall_bar(df, "rmse", out, dpi=72)
        assert out.exists()


class TestPlotMetricByCount:
    def test_creates_png(self, tmp_path):
        df = pd.DataFrame({"n_feats": [1, 1, 2, 2], "rmse": [0.5, 0.6, 0.4, 0.3]})
        out = tmp_path / "count.png"
        plot_metric_by_count(df, "rmse", out, dpi=72)
        assert out.exists()


class TestPlotMarginalBox:
    def test_creates_png(self, tmp_path):
        df = pd.DataFrame({
            "feature": ["era5"] * 5 + ["hrrr"] * 5,
            "delta": list(np.random.default_rng(0).normal(0, 0.01, 10)),
        })
        out = tmp_path / "box.png"
        plot_marginal_box(df, "rmse", out, config=WIND_PLOT_CONFIG, dpi=72)
        assert out.exists()


class TestPlotMarginalBarMeans:
    def test_creates_png(self, tmp_path):
        df = pd.DataFrame({
            "feature": ["era5"] * 4 + ["hrrr"] * 4,
            "delta": list(np.random.default_rng(1).normal(0, 0.01, 8)),
        })
        out = tmp_path / "bar_means.png"
        plot_marginal_bar_means(df, "rmse", out,
                                feature_list=["era5", "hrrr"], dpi=72)
        assert out.exists()

    def test_empty(self, tmp_path):
        df = pd.DataFrame(columns=["feature", "delta"])
        out = tmp_path / "bar_means.png"
        plot_marginal_bar_means(df, "rmse", out, feature_list=["era5"], dpi=72)
        # Should not create file when empty
        assert not out.exists()


class TestPlotComboPresenceHeatmap:
    def test_colored_mode(self, tmp_path):
        df = pd.DataFrame({
            "label": ["era5+hrrr", "hrrr"],
            "feats": [("era5", "hrrr"), ("hrrr",)],
            "mae": [0.5, 0.6],
        })
        out = tmp_path / "heatmap.png"
        plot_combo_presence_heatmap(
            df, out, feature_list=["era5", "hrrr"],
            config=WIND_PLOT_CONFIG, dpi=72,
        )
        assert out.exists()

    def test_grayscale_mode(self, tmp_path):
        df = pd.DataFrame({
            "label": ["latlon+height", "height"],
            "feats": [("latlon", "height"), ("height",)],
            "rmse": [0.5, 0.6],
        })
        out = tmp_path / "heatmap_gs.png"
        plot_combo_presence_heatmap(
            df, out, feature_list=["latlon", "height"],
            config=AUX_PLOT_CONFIG, key_metric="rmse", dpi=72,
        )
        assert out.exists()

    def test_empty_df(self, tmp_path):
        df = pd.DataFrame(columns=["label", "feats", "mae"])
        out = tmp_path / "heatmap_empty.png"
        plot_combo_presence_heatmap(
            df, out, feature_list=["era5"], config=WIND_PLOT_CONFIG, dpi=72,
        )
        # Should not create file when empty
        assert not out.exists()


# ===================== integration =====================

class TestAnalyzeSweepResults:
    @pytest.fixture
    def sweep_dir(self, tmp_path):
        """Create a minimal sweep directory with synthetic result CSVs."""
        rng = np.random.default_rng(42)
        features = ["era5", "hrrr"]

        # Create combos: era5, hrrr, era5+hrrr
        for label in ["era5", "hrrr", "era5+hrrr"]:
            n = 20
            obs = rng.uniform(3, 10, n)
            pred = obs + rng.normal(0, 0.5, n)
            df = pd.DataFrame({
                "station_id": [f"S{i % 3}" for i in range(n)],
                "observation_type": ["GS"] * n,
                "observation": obs,
                "pred_observation": pred,
            })
            df.to_csv(tmp_path / f"ml_results_xgb_{label}.csv", index=False)

        return tmp_path

    def test_produces_expected_files(self, sweep_dir):
        config = {
            "feature_labels": {"era5": "ERA5", "hrrr": "HRRR"},
            "box_order": ["ERA5", "HRRR"],
            "box_palette": ["#4E79A7", "#E15759"],
            "heatmap_mode": "colored",
            "heatmap_palette": {"ERA5": "#4E79A7", "HRRR": "#E15759"},
            "metric_xlabel": "# features used",
            "file_prefix": "ml_results_xgb_",
        }

        outdir = analyze_sweep_results(
            results_dir=sweep_dir,
            feature_list=["era5", "hrrr"],
            config=config,
            key_metric="rmse",
            top_k=5,
            dpi=72,
        )

        assert outdir.exists()
        assert (outdir / "metrics_summary.csv").exists()
        assert (outdir / "marginal_deltas.csv").exists()
        assert (outdir / "overall_rmse_by_combo.png").exists()
        assert (outdir / "rmse_by_num_features.png").exists()
        assert (outdir / "marginal_delta_rmse_boxplot.png").exists()
        assert (outdir / "combo_presence_heatmap_topN.png").exists()
