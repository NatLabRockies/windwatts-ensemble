"""Tests for wem.analyze.quantile_maps."""

import numpy as np
import pandas as pd
import pytest

from wem.analyze.quantile_maps import (
    build_bias_df,
    choose_col,
    filter_by_bias,
    find_qcols,
    load_gwa_file,
    load_quantile_file,
    mean_from_quantiles,  # re-exported from wem.utils.quantiles
)


# ---- choose_col ----

class TestChooseCol:
    def test_exact_match(self):
        df = pd.DataFrame({"station_id": [1], "lat": [40.0]})
        assert choose_col(df, ["station_id"]) == "station_id"

    def test_case_insensitive(self):
        df = pd.DataFrame({"STATION_ID": [1], "LAT": [40.0]})
        assert choose_col(df, ["station_id"]) == "STATION_ID"

    def test_no_match(self):
        df = pd.DataFrame({"foo": [1]})
        assert choose_col(df, ["bar", "baz"]) is None


# ---- find_qcols ----

class TestFindQcols:
    def test_full_set(self):
        cols = {f"q{i:03d}": [float(i)] for i in range(101)}
        cols["station_id"] = ["A"]
        df = pd.DataFrame(cols)
        qcols = find_qcols(df)
        assert len(qcols) == 101
        assert qcols[0] == "q000"
        assert qcols[-1] == "q100"

    def test_partial_set(self):
        cols = {f"q{i:03d}": [float(i)] for i in range(0, 101, 10)}
        df = pd.DataFrame(cols)
        qcols = find_qcols(df)
        assert len(qcols) == 11

    def test_no_quantile_cols(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        qcols = find_qcols(df)
        assert len(qcols) == 0


# ---- mean_from_quantiles ----

class TestMeanFromQuantiles:
    def test_linear(self):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {c: np.linspace(0, 10, 101)[i] for i, c in enumerate(qcols)}
        row = pd.Series(data)
        result = mean_from_quantiles(row, qcols)
        assert result is not None
        assert abs(result - 5.0) < 0.1

    def test_constant(self):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {c: 3.0 for c in qcols}
        row = pd.Series(data)
        result = mean_from_quantiles(row, qcols)
        assert result is not None
        assert abs(result - 3.0) < 0.01

    def test_insufficient_data(self):
        row = pd.Series({"q000": 5.0})
        result = mean_from_quantiles(row, ["q000"])
        assert result is None

    def test_nan_handling(self):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {c: np.linspace(0, 10, 101)[i] for i, c in enumerate(qcols)}
        data["q050"] = np.nan
        row = pd.Series(data)
        result = mean_from_quantiles(row, qcols)
        assert result is not None
        assert np.isfinite(result)


# ---- load_quantile_file ----

class TestLoadQuantileFile:
    def test_happy_path(self, tmp_path):
        qcols = [f"q{i:03d}" for i in range(101)]
        data = {
            "station_id": ["A", "B"],
            "name": ["Site A", "Site B"],
            "lat": [40.0, 41.0],
            "lon": [-100.0, -101.0],
        }
        for c in qcols:
            data[c] = [3.0, 4.0]
        df = pd.DataFrame(data)
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)

        result = load_quantile_file(p, "Test")
        assert "mean_ws" in result.columns
        assert len(result) == 2
        assert abs(result["mean_ws"].iloc[0] - 3.0) < 0.01

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_quantile_file(tmp_path / "missing.csv", "Test")

    def test_missing_columns(self, tmp_path):
        df = pd.DataFrame({"x": [1], "y": [2]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="required columns"):
            load_quantile_file(p, "Test")


# ---- load_gwa_file ----

class TestLoadGwaFile:
    def test_happy_path(self, tmp_path):
        data = {
            "station_id": ["A"],
            "lat": [40.0],
            "lon": [-100.0],
            "height_m": [10.0],
            "gwa_interp": [5.5],
        }
        p = tmp_path / "gwa.csv"
        pd.DataFrame(data).to_csv(p, index=False)
        result = load_gwa_file(p, "Test")
        assert len(result) == 1
        assert abs(result["mean_ws"].iloc[0] - 5.5) < 0.01

    def test_none_path(self):
        result = load_gwa_file(None, "Test")
        assert result.empty

    def test_missing_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_gwa_file(p, "Test")


# ---- build_bias_df ----

class TestBuildBiasDf:
    def test_basic_merge(self):
        model = pd.DataFrame({
            "station_id": ["A", "B"],
            "lat": [40.0, 41.0],
            "lon": [-100.0, -101.0],
            "name": ["a", "b"],
            "mean_ws": [5.0, 6.0],
        })
        obs = pd.DataFrame({
            "station_id": ["A", "B"],
            "lat": [40.0, 41.0],
            "lon": [-100.0, -101.0],
            "name": ["a", "b"],
            "mean_ws": [4.5, 5.5],
        })
        result = build_bias_df(model, obs, "Test", ["station_id"])
        assert len(result) == 2
        assert abs(result["bias"].iloc[0] - 0.5) < 0.01
        assert abs(result["bias"].iloc[1] - 0.5) < 0.01

    def test_no_overlap(self):
        model = pd.DataFrame({
            "station_id": ["A"],
            "lat": [40.0],
            "lon": [-100.0],
            "name": ["a"],
            "mean_ws": [5.0],
        })
        obs = pd.DataFrame({
            "station_id": ["C"],
            "lat": [42.0],
            "lon": [-102.0],
            "name": ["c"],
            "mean_ws": [4.0],
        })
        result = build_bias_df(model, obs, "Test", ["station_id"])
        assert result.empty

    def test_height_join(self):
        model = pd.DataFrame({
            "station_id": ["A", "A"],
            "height_m": [10.0, 80.0],
            "lat": [40.0, 40.0],
            "lon": [-100.0, -100.0],
            "name": ["a", "a"],
            "mean_ws": [3.0, 5.0],
        })
        obs = pd.DataFrame({
            "station_id": ["A", "A"],
            "height_m": [10.0, 80.0],
            "lat": [40.0, 40.0],
            "lon": [-100.0, -100.0],
            "name": ["a", "a"],
            "mean_ws": [2.5, 4.5],
        })
        result = build_bias_df(model, obs, "Test", ["station_id", "height_m"])
        assert len(result) == 2
        assert abs(result["bias"].iloc[0] - 0.5) < 0.01


# ---- filter_by_bias ----

class TestFilterByBias:
    def test_removes_outliers(self):
        obs = pd.DataFrame({
            "station_id": ["A", "B", "C"],
            "lat": [40.0, 41.0, 42.0],
            "lon": [-100.0, -101.0, -102.0],
            "name": ["a", "b", "c"],
            "mean_ws": [5.0, 5.0, 5.0],
        })
        era5 = pd.DataFrame({
            "station_id": ["A", "B", "C"],
            "lat": [40.0, 41.0, 42.0],
            "lon": [-100.0, -101.0, -102.0],
            "name": ["a", "b", "c"],
            "mean_ws": [5.1, 5.2, 100.0],  # C is a huge outlier
        })
        keep = filter_by_bias(obs, era5, bias_trim=0.02)
        assert isinstance(keep, set)
        assert "A" in keep
        assert "B" in keep
        assert "C" not in keep

    def test_keeps_inliers(self):
        # Use enough stations so none are trimmed
        n = 20
        obs = pd.DataFrame({
            "station_id": [f"S{i}" for i in range(n)],
            "lat": [40.0 + i * 0.1 for i in range(n)],
            "lon": [-100.0 - i * 0.1 for i in range(n)],
            "name": [f"s{i}" for i in range(n)],
            "mean_ws": [5.0] * n,
        })
        era5 = pd.DataFrame({
            "station_id": [f"S{i}" for i in range(n)],
            "lat": [40.0 + i * 0.1 for i in range(n)],
            "lon": [-100.0 - i * 0.1 for i in range(n)],
            "name": [f"s{i}" for i in range(n)],
            "mean_ws": [5.1] * n,
        })
        keep = filter_by_bias(obs, era5, bias_trim=0.02)
        assert len(keep) == n

    def test_returns_set(self):
        obs = pd.DataFrame({
            "station_id": ["A"],
            "lat": [40.0],
            "lon": [-100.0],
            "name": ["a"],
            "mean_ws": [5.0],
        })
        era5 = pd.DataFrame({
            "station_id": ["A"],
            "lat": [40.0],
            "lon": [-100.0],
            "name": ["a"],
            "mean_ws": [5.5],
        })
        result = filter_by_bias(obs, era5, bias_trim=0.02)
        assert isinstance(result, set)
