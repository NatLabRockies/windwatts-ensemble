"""Data transform functions for experiment types.

- ``enrich_with_cdf`` — adds full CDF features for the enriched experiment
- ``enrich_with_cdf_subset`` — adds selected quantile CDF features (e.g. q50, q90)
- ``pivot_to_wide`` — long → wide pivot for the wide experiment
- ``wide_preds_to_long`` — converts wide predictions back to long format
- ``wide_to_convnet_arrays`` — extracts CDF/aux/target numpy arrays for convnet
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def enrich_with_cdf(
    df: pd.DataFrame,
    wind_cols: list[str],
) -> pd.DataFrame:
    """Add full CDF features for each wind column to every row.

    For each ``(station_id, height_m)`` group (which has 101 rows, one per
    quantile), pivots each wind column's values across quantiles and merges
    them back so every row gains ``{wc}_q000`` ... ``{wc}_q100`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format table with one row per (station_id, height_m, qnum).
    wind_cols : list[str]
        Wind resource column names to pivot (e.g. ``["hrrr", "wtk"]``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 101 extra columns per wind column.
        Row count is preserved.
    """
    group_keys = ["station_id", "height_m"]
    result = df.copy()

    for wc in wind_cols:
        if wc not in result.columns:
            continue
        sub = result[group_keys + ["qnum", wc]].copy()
        piv = sub.pivot_table(
            index=group_keys, columns="qnum", values=wc, aggfunc="first",
        )
        piv.columns = [f"{wc}_q{int(q):03d}" for q in piv.columns]
        piv = piv.reset_index()
        result = result.merge(piv, on=group_keys, how="left")

    return result


def enrich_with_cdf_subset(
    df: pd.DataFrame,
    wind_cols: list[str],
    quantiles: list[int] | None = None,
) -> pd.DataFrame:
    """Add selected quantile CDF features for each wind column to every row.

    Lightweight version of :func:`enrich_with_cdf` that pivots only specific
    quantile values instead of all 101.  Useful for providing CDF shape context
    (e.g. q50 and q90) without adding hundreds of columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format table with one row per (station_id, height_m, qnum).
    wind_cols : list[str]
        Wind resource column names to pivot (e.g. ``["hrrr", "wtk"]``).
    quantiles : list[int] or None
        Quantile indices to extract (default ``[50, 90]``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``len(quantiles)`` extra columns per wind column.
        Column naming: ``{wc}_q{NN:03d}`` (e.g. ``hrrr_q050``).
        Row count is preserved.
    """
    if quantiles is None:
        quantiles = [50, 90]

    group_keys = ["station_id", "height_m"]
    result = df.copy()

    for wc in wind_cols:
        if wc not in result.columns:
            continue
        sub = result.loc[result["qnum"].isin(quantiles), group_keys + ["qnum", wc]].copy()
        piv = sub.pivot_table(
            index=group_keys, columns="qnum", values=wc, aggfunc="first",
        )
        piv.columns = [f"{wc}_q{int(q):03d}" for q in piv.columns]
        piv = piv.reset_index()
        result = result.merge(piv, on=group_keys, how="left")

    return result


def pivot_to_wide(
    df: pd.DataFrame,
    wind_cols: list[str],
    target_col: str = "observation",
) -> pd.DataFrame:
    """Pivot long-format (station, height, qnum) → wide (station, height).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format training table with one row per (station_id, height_m, qnum).
        Must contain *wind_cols*, *target_col*, ``qnum``, ``station_id``,
        ``height_m``, and auxiliary columns.
    wind_cols : list[str]
        Wind resource column names to pivot (e.g. ``["hrrr", "wtk", ...]``).
    target_col : str
        Name of the observation/target column to pivot.

    Returns
    -------
    pd.DataFrame
        Wide-format table with columns:
        ``station_id``, ``height_m``, auxiliary cols,
        ``{wc}_q000``...``{wc}_q100`` per wind col,
        ``obs_q000``...``obs_q100`` for the target.
        Groups with != 101 quantile rows are dropped.
    """
    required = ["station_id", "height_m", "qnum", target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    aux_candidates = [
        "lat", "lon", "elevation_m", "height_m", "observation_type",
        "gwa_interp", "neighbors_10km_site_ids",
    ]
    aux_cols = [c for c in aux_candidates if c in df.columns and c != "height_m"]

    group_keys = ["station_id", "height_m"]
    groups = df.groupby(group_keys, sort=False)

    valid_groups = groups.filter(lambda g: len(g) == 101)
    if valid_groups.empty:
        raise ValueError("No station-height groups with exactly 101 quantile rows")

    groups = valid_groups.groupby(group_keys, sort=False)
    aux_df = groups[aux_cols].first().reset_index()

    wide_df = aux_df

    for wc in wind_cols:
        if wc not in df.columns:
            continue
        sub = valid_groups[group_keys + ["qnum", wc]].copy()
        piv = sub.pivot_table(
            index=group_keys, columns="qnum", values=wc, aggfunc="first",
        )
        piv.columns = [f"{wc}_q{int(q):03d}" for q in piv.columns]
        wide_df = wide_df.merge(piv, on=group_keys)

    sub_obs = valid_groups[group_keys + ["qnum", target_col]].copy()
    piv_obs = sub_obs.pivot_table(
        index=group_keys, columns="qnum", values=target_col, aggfunc="first",
    )
    piv_obs.columns = [f"obs_q{int(q):03d}" for q in piv_obs.columns]
    wide_df = wide_df.merge(piv_obs, on=group_keys)

    return wide_df


def wide_preds_to_long(
    wide_df: pd.DataFrame,
    preds_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Convert wide-format predictions to long format matching ``ml_results.csv``.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide-format table with ``station_id``, ``height_m``, ``obs_q000``...``obs_q100``.
    preds_dict : dict
        ``{station_id: (test_indices_into_wide_df, preds_101_array)}``.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns: ``station_id``, ``height_m``,
        ``observation_type``, ``qnum``, ``observation``, ``pred_observation``.
    """
    rows = []
    obs_cols = [f"obs_q{q:03d}" for q in range(101)]

    for sid, (test_idx, preds_arr) in preds_dict.items():
        if test_idx.size == 0:
            continue
        for i, idx in enumerate(test_idx):
            row_data = wide_df.iloc[idx]
            station_id = row_data["station_id"]
            height_m = row_data["height_m"]
            obs_type = row_data.get("observation_type", "")

            for q in range(101):
                obs_val = row_data[obs_cols[q]] if obs_cols[q] in row_data.index else np.nan
                pred_val = float(preds_arr[i, q])
                rows.append({
                    "station_id": station_id,
                    "height_m": height_m,
                    "observation_type": obs_type,
                    "qnum": q,
                    "observation": obs_val,
                    "pred_observation": pred_val,
                })

    return pd.DataFrame(rows)


def wide_to_convnet_arrays(
    wide_df: pd.DataFrame,
    wind_cols: List[str],
    include_gwa: bool = False,
    gwa_feature_name: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract CDF, auxiliary, and target arrays from a wide-format DataFrame.

    Converts the wide table (one row per station-height) into the numpy arrays
    needed by the ConvNet fold worker. No torch dependency — pure numpy/pandas.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide-format table from ``pivot_to_wide``.
    wind_cols : list[str]
        Wind resource column names (e.g. ``["hrrr", "wtk", "wtk_led_conus"]``).
    include_gwa : bool
        Whether to include GWA as an auxiliary feature.
    gwa_feature_name : str or None
        Name of the GWA column (e.g. ``"gwa_interp"``).

    Returns
    -------
    cdf_array : np.ndarray
        (N, n_channels, 101) float32 — one channel per wind column.
    aux_array : np.ndarray
        (N, n_aux) float32 — auxiliary features.
    target_array : np.ndarray
        (N, 101) float32 — observed CDF targets.
    """
    n = len(wide_df)
    n_channels = len(wind_cols)

    # CDF channels: (N, n_channels, 101)
    cdf_array = np.empty((n, n_channels, 101), dtype=np.float32)
    for ch, wc in enumerate(wind_cols):
        cdf_cols = [f"{wc}_q{q:03d}" for q in range(101)]
        cdf_array[:, ch, :] = wide_df[cdf_cols].to_numpy(dtype=np.float32)

    # Auxiliary features
    aux_cols = [c for c in ["lat", "lon", "height_m", "elevation_m"] if c in wide_df.columns]
    if include_gwa and gwa_feature_name and gwa_feature_name in wide_df.columns:
        aux_cols.append(gwa_feature_name)
    aux_array = wide_df[aux_cols].to_numpy(dtype=np.float32)

    # Target: obs_q000..obs_q100
    obs_cols = [f"obs_q{q:03d}" for q in range(101)]
    target_array = wide_df[obs_cols].to_numpy(dtype=np.float32)

    return cdf_array, aux_array, target_array
