"""Tests for wem.experiment.transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wem.experiment.transforms import (
    enrich_with_cdf,
    enrich_with_cdf_subset,
    pivot_to_wide,
    wide_preds_to_long,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic data
# ---------------------------------------------------------------------------


def _make_long_df(
    n_stations: int = 4,
    n_heights: int = 2,
    wind_cols: list[str] | None = None,
    obs_type_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic long-format training table.

    Returns a DataFrame with ``n_stations * n_heights * 101`` rows.
    """
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


# ===================== TestEnrichWithCdf =====================


class TestEnrichWithCdf:
    def test_columns_added(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        enriched = enrich_with_cdf(df, ["hrrr"])
        for q in [0, 50, 100]:
            assert f"hrrr_q{q:03d}" in enriched.columns

    def test_preserves_row_count(self):
        df = _make_long_df(n_stations=3, n_heights=2, wind_cols=["hrrr", "wtk"])
        enriched = enrich_with_cdf(df, ["hrrr", "wtk"])
        assert len(enriched) == len(df)

    def test_cdf_values_consistent(self):
        """For a given station/height, hrrr_q{N} should equal hrrr at qnum=N."""
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        enriched = enrich_with_cdf(df, ["hrrr"])

        grp = enriched[enriched["station_id"] == "S000"]
        for _, row in grp.iterrows():
            q = int(row["qnum"])
            orig_val = row["hrrr"]
            cdf_val = row[f"hrrr_q{q:03d}"]
            assert abs(orig_val - cdf_val) < 1e-10

    def test_cdf_values_replicated_across_qnums(self):
        """All rows in the same station/height group should have identical CDF values."""
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        enriched = enrich_with_cdf(df, ["hrrr"])

        grp = enriched[enriched["station_id"] == "S000"]
        # hrrr_q050 should be the same for every row in this group
        vals = grp["hrrr_q050"].unique()
        assert len(vals) == 1

    def test_multiple_wind_cols(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr", "wtk"])
        enriched = enrich_with_cdf(df, ["hrrr", "wtk"])
        assert "hrrr_q050" in enriched.columns
        assert "wtk_q050" in enriched.columns
        # 101 columns per wind source = 202 new columns
        new_cols = [c for c in enriched.columns if c not in df.columns]
        assert len(new_cols) == 2 * 101

    def test_missing_wind_col_skipped(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        enriched = enrich_with_cdf(df, ["hrrr", "nonexistent"])
        assert "hrrr_q050" in enriched.columns
        assert "nonexistent_q050" not in enriched.columns

    def test_original_columns_preserved(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        enriched = enrich_with_cdf(df, ["hrrr"])
        # Original hrrr column should still be present
        assert "hrrr" in enriched.columns
        assert "station_id" in enriched.columns
        assert "qnum" in enriched.columns


# ===================== TestEnrichWithCdfSubset =====================


class TestEnrichWithCdfSubset:
    def test_column_count(self):
        """Default quantiles=[50,90] with 2 wind cols → 4 new columns."""
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr", "wtk"])
        enriched = enrich_with_cdf_subset(df, ["hrrr", "wtk"])
        new_cols = [c for c in enriched.columns if c not in df.columns]
        assert len(new_cols) == 4  # hrrr_q050, hrrr_q090, wtk_q050, wtk_q090

    def test_value_correctness(self):
        """CDF subset values should match the original wind column at that quantile."""
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        enriched = enrich_with_cdf_subset(df, ["hrrr"])

        grp = enriched[enriched["station_id"] == "S000"]
        # hrrr_q050 should equal the hrrr value at qnum=50 for this station
        row_q50 = grp[grp["qnum"] == 50].iloc[0]
        assert abs(row_q50["hrrr_q050"] - row_q50["hrrr"]) < 1e-10

    def test_broadcast_across_qnums(self):
        """All rows in same station/height group should have same CDF subset values."""
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        enriched = enrich_with_cdf_subset(df, ["hrrr"])

        grp = enriched[enriched["station_id"] == "S000"]
        vals = grp["hrrr_q050"].unique()
        assert len(vals) == 1

    def test_custom_quantile_list(self):
        """Custom quantiles should produce the right columns."""
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr", "wtk"])
        enriched = enrich_with_cdf_subset(df, ["hrrr", "wtk"], quantiles=[25, 75])
        assert "hrrr_q025" in enriched.columns
        assert "hrrr_q075" in enriched.columns
        assert "wtk_q025" in enriched.columns
        assert "wtk_q075" in enriched.columns
        assert "hrrr_q050" not in enriched.columns  # not requested


# ===================== TestPivotToWide =====================


class TestPivotToWide:
    def test_shape(self):
        df = _make_long_df(n_stations=3, n_heights=2, wind_cols=["hrrr", "wtk"])
        wide = pivot_to_wide(df, ["hrrr", "wtk"])
        # 3 stations x 2 heights = 6 wide rows
        assert wide.shape[0] == 6

    def test_feature_columns(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])
        for q in [0, 50, 100]:
            assert f"hrrr_q{q:03d}" in wide.columns

    def test_target_columns(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])
        for q in [0, 50, 100]:
            assert f"obs_q{q:03d}" in wide.columns

    def test_incomplete_group_dropped(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        # Remove some quantile rows from station S000
        mask = ~((df["station_id"] == "S000") & (df["qnum"] >= 90))
        df_partial = df[mask].reset_index(drop=True)
        wide = pivot_to_wide(df_partial, ["hrrr"])
        # S000 should be dropped, only S001 remains
        assert wide.shape[0] == 1
        assert wide["station_id"].iloc[0] == "S001"

    def test_aux_columns_carried(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])
        assert "lat" in wide.columns
        assert "lon" in wide.columns
        assert "elevation_m" in wide.columns
        assert "observation_type" in wide.columns

    def test_missing_column_raises(self):
        df = _make_long_df(n_stations=2, n_heights=1)
        df = df.drop(columns=["qnum"])
        with pytest.raises(ValueError, match="Missing required columns"):
            pivot_to_wide(df, ["hrrr"])

    def test_multiple_wind_cols(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr", "wtk", "wtk_led_conus"])
        wide = pivot_to_wide(df, ["hrrr", "wtk", "wtk_led_conus"])
        for wc in ["hrrr", "wtk", "wtk_led_conus"]:
            assert f"{wc}_q050" in wide.columns


# ===================== TestWidePredsToLong =====================


class TestWidePredsToLong:
    def test_schema(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])

        preds = np.arange(101, dtype=np.float32).reshape(1, 101) * 0.1
        preds_dict = {"S000": (np.array([0]), preds)}

        long = wide_preds_to_long(wide, preds_dict)
        assert "station_id" in long.columns
        assert "height_m" in long.columns
        assert "qnum" in long.columns
        assert "observation" in long.columns
        assert "pred_observation" in long.columns
        assert "observation_type" in long.columns

    def test_qnum_range(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])

        preds = np.ones((1, 101), dtype=np.float32)
        preds_dict = {"S000": (np.array([0]), preds)}

        long = wide_preds_to_long(wide, preds_dict)
        assert set(long["qnum"].unique()) == set(range(101))

    def test_correct_row_count(self):
        df = _make_long_df(n_stations=3, n_heights=2, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])

        preds_dict = {
            "S000": (np.array([0, 1]), np.ones((2, 101), dtype=np.float32)),
            "S001": (np.array([2, 3]), np.ones((2, 101), dtype=np.float32)),
        }
        long = wide_preds_to_long(wide, preds_dict)
        # 2 stations x 2 heights x 101 quantiles = 404 rows
        assert len(long) == 4 * 101

    def test_empty_preds_dict(self):
        df = _make_long_df(n_stations=2, n_heights=1, wind_cols=["hrrr"])
        wide = pivot_to_wide(df, ["hrrr"])

        long = wide_preds_to_long(wide, {})
        assert len(long) == 0
