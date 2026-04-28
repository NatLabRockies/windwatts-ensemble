#!/usr/bin/env python3
"""Unified experiment runner for WEM LOOCV experiments.

CLI: ``wem-experiment <type> [args]``

Experiment types:

  baseline  — Production long-format single-model LOOCV (9 features)
  enriched  — Long-format + full CDF features (309 features)
  wide      — Wide-format multi-output (101 independent models, 308 features)
  convnet   — 1D Conv-Net CDF-to-CDF (3-channel input, monotonic output)
  mlp       — Long-format MLP (same features as baseline, PyTorch)
  hybrid    — Baseline q0-q94, log-target tail-specialized model q95-q100
  compare   — Compare experiment results

Examples::

    wem-experiment baseline --infile data.csv --outfile results.csv --n-jobs 12
    wem-experiment enriched --infile data.csv --outfile enriched.csv --stations "S1,S2"
    wem-experiment wide --infile data.csv --outfile wide.csv
    wem-experiment convnet --infile data.csv --outfile convnet.csv --epochs 300 --device cpu
    wem-experiment mlp --infile data.csv --outfile mlp.csv --epochs 100 --device mps
    wem-experiment hybrid --infile data.csv --outfile hybrid.csv --tail-cutoff 95
    wem-experiment compare --baseline base.csv --experiments enriched.csv --labels Enriched
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise SystemExit(
        "This script requires xgboost. Install with `pip install xgboost`"
    ) from e

from joblib import Parallel, delayed

from wem.constants import DEFAULT_XGB_PARAMS, WIND_FEATURE_MAP
from wem.experiment.transforms import (
    enrich_with_cdf,
    enrich_with_cdf_subset,
    pivot_to_wide,
    wide_preds_to_long,
    wide_to_convnet_arrays,
)
from wem.train.loocv_xgb import run_one_fold
from wem.utils.logging import log
from wem.utils.ml import (
    balance_indices,
    build_neighbor_map,
    fold_seed,
    make_features,
    merge_gwa_feature,
)
from wem.utils.sites import normalize_obs_type


# Default auxiliary feature columns
AUX_COLS = ["lat", "lon", "height_m", "elevation_m"]


# ---------------------------------------------------------------------------
# Experiment type registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentType:
    """Describes an experiment formulation."""

    name: str
    description: str
    format: str  # "long" or "wide"


EXPERIMENTS: dict[str, ExperimentType] = {
    "baseline": ExperimentType(
        name="baseline",
        description="Production long-format single-model LOOCV (9 features)",
        format="long",
    ),
    "enriched": ExperimentType(
        name="enriched",
        description="Long-format + full CDF features (309 features)",
        format="long",
    ),
    "wide": ExperimentType(
        name="wide",
        description="Wide-format multi-output (101 independent models, 308 features)",
        format="wide",
    ),
    "convnet": ExperimentType(
        name="convnet",
        description="1D Conv-Net CDF-to-CDF (3-channel input, monotonic output)",
        format="convnet",
    ),
    "mlp": ExperimentType(
        name="mlp",
        description="Long-format MLP (same features as baseline, PyTorch)",
        format="mlp",
    ),
    "hybrid": ExperimentType(
        name="hybrid",
        description="Hybrid: baseline q0-q94, log-target tail model q95-q100",
        format="hybrid",
    ),
    "cdf_context": ExperimentType(
        name="cdf_context",
        description="Baseline + CDF context features (q50/q90 per wind source, 15 features)",
        format="long",
    ),
}


# ---------------------------------------------------------------------------
# Wide-format fold worker (moved from multi_output.py)
# ---------------------------------------------------------------------------


def run_one_fold_wide(
    sid: str,
    X_wide: np.ndarray,
    Y_wide: np.ndarray,
    station_ids: np.ndarray,
    is_gs: np.ndarray,
    nbr_map: Dict[str, Set[str]],
    xgb_params: dict,
    seed: int,
    balance_strategy: str = "downsample",
) -> Tuple[str, np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
    """Train 101 XGBoost models for one LOOCV fold and predict the test station's CDFs.

    Parameters
    ----------
    sid : str
        Station ID for the held-out test station.
    X_wide : np.ndarray
        (N, F) feature array in wide format.
    Y_wide : np.ndarray
        (N, 101) target array — one column per quantile.
    station_ids : np.ndarray
        (N,) station IDs aligned with X_wide rows.
    is_gs : np.ndarray
        (N,) boolean mask for Gold Standard stations.
    nbr_map : dict
        Neighbor map from ``build_neighbor_map``.
    xgb_params : dict
        XGBoost hyperparameters.
    seed : int
        Base random seed.
    balance_strategy : str
        ``"downsample"`` or ``"upsample"``.

    Returns
    -------
    tuple
        ``(sid, test_indices, predictions_101, metrics_or_None)``
    """
    n_quantiles = Y_wide.shape[1]

    # Test rows = this GS station
    test_idx = np.where(station_ids == sid)[0]
    if test_idx.size == 0:
        return sid, np.array([], dtype=int), np.empty((0, n_quantiles), dtype=np.float32), None

    # Exclusion set: this sid + its 10km neighbors
    excl: Set[str] = {sid}
    if sid in nbr_map:
        excl |= set(nbr_map[sid])

    # Base training mask: not excluded, finite target on all quantiles
    excl_arr = np.fromiter(excl, dtype=station_ids.dtype)
    base_train_mask = ~np.isin(station_ids, excl_arr) & np.all(np.isfinite(Y_wide), axis=1)

    base_idx = np.where(base_train_mask)[0]
    if base_idx.size == 0:
        return sid, test_idx, np.full((test_idx.size, n_quantiles), np.nan, dtype=np.float32), None

    # Require finite features
    finite_feat_mask = np.all(np.isfinite(X_wide[base_idx]), axis=1)
    good_train_idx = base_idx[finite_feat_mask]
    if good_train_idx.size < 10:
        return sid, test_idx, np.full((test_idx.size, n_quantiles), np.nan, dtype=np.float32), None

    # Balance GS/ASOS
    idx_asos = good_train_idx[~is_gs[good_train_idx]]
    idx_gs = good_train_idx[is_gs[good_train_idx]]

    rng = np.random.default_rng(fold_seed(seed, sid))
    train_idx = balance_indices(idx_asos, idx_gs, rng, strategy=balance_strategy)
    if train_idx.size < 10:
        return sid, test_idx, np.full((test_idx.size, n_quantiles), np.nan, dtype=np.float32), None

    rng.shuffle(train_idx)

    X_train = X_wide[train_idx]
    Y_train = Y_wide[train_idx]
    X_test = X_wide[test_idx]

    # Train 101 independent models (one per quantile target)
    preds = np.empty((test_idx.size, n_quantiles), dtype=np.float32)
    for q in range(n_quantiles):
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, Y_train[:, q])
        preds[:, q] = model.predict(X_test).astype(np.float32)

    # Enforce monotonicity across quantiles via cumulative max
    preds = np.maximum.accumulate(preds, axis=1)

    # Compute per-station metrics (flat across all quantiles)
    Y_test = Y_wide[test_idx]
    good = np.isfinite(Y_test) & np.isfinite(preds)
    metrics = None
    if np.any(good):
        diff = preds[good] - Y_test[good]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        metrics = (rmse, mae)

    return sid, test_idx, preds, metrics


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------


def build_long_features(
    df: pd.DataFrame,
    wind_cols: list[str],
    include_gwa: bool,
    gwa_feature_name: Optional[str],
    enriched: bool = False,
) -> Tuple[list[str], str, np.ndarray]:
    """Build feature columns, monotonic constraints, and finite mask for long-format experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame (with CDF columns if *enriched* is True).
    wind_cols : list[str]
        Wind resource column names.
    include_gwa : bool
        Whether to include GWA as a feature.
    gwa_feature_name : str or None
        Name of the GWA feature column.
    enriched : bool
        If True, use CDF columns instead of raw wind columns.

    Returns
    -------
    feat_cols : list[str]
        Ordered feature column names.
    mon_str : str
        Monotonic constraints string for XGBoost.
    require_finite : np.ndarray
        Boolean mask — True for columns that require finite values.
    """
    feat_cols: list[str] = ["qnum"]

    if enriched:
        # CDF features: {wc}_q000..{wc}_q100 for each wind col
        for wc in wind_cols:
            feat_cols.extend([f"{wc}_q{q:03d}" for q in range(101)])
    else:
        # Baseline: raw wind columns
        feat_cols.extend(wind_cols)

    # Auxiliary features
    feat_cols.extend([c for c in AUX_COLS if c in df.columns])

    # GWA
    if include_gwa and gwa_feature_name and gwa_feature_name in df.columns:
        feat_cols.append(gwa_feature_name)

    # Filter to columns that exist
    feat_cols = [c for c in feat_cols if c in df.columns]

    # Monotonic constraint: +1 on qnum, 0 on all others
    mon = [0] * len(feat_cols)
    if "qnum" in feat_cols:
        mon[feat_cols.index("qnum")] = 1
    mon_str = "(" + ",".join(map(str, mon)) + ")"

    # Require finite mask: all except GWA
    require_finite = np.ones(len(feat_cols), dtype=bool)
    if include_gwa and gwa_feature_name and gwa_feature_name in feat_cols:
        require_finite[feat_cols.index(gwa_feature_name)] = False

    return feat_cols, mon_str, require_finite


def build_wide_features(
    wide_df: pd.DataFrame,
    wind_cols: list[str],
    include_gwa: bool,
    gwa_feature_name: Optional[str],
) -> Tuple[list[str], list[str]]:
    """Build feature and target columns for wide-format experiments.

    Returns
    -------
    feat_cols : list[str]
        Ordered feature column names.
    obs_cols : list[str]
        Target column names (``obs_q000``...``obs_q100``).
    """
    feat_cols: list[str] = []

    # CDF features per wind column
    for wc in wind_cols:
        feat_cols.extend([f"{wc}_q{q:03d}" for q in range(101)])

    # Auxiliary
    feat_cols.extend([c for c in AUX_COLS if c in wide_df.columns])

    # GWA
    if include_gwa and gwa_feature_name and gwa_feature_name in wide_df.columns:
        feat_cols.append(gwa_feature_name)

    feat_cols = [c for c in feat_cols if c in wide_df.columns]

    obs_cols = [f"obs_q{q:03d}" for q in range(101)]
    return feat_cols, obs_cols


def build_hybrid_tail_features(
    df: pd.DataFrame,
    wind_cols: list[str],
    include_gwa: bool,
    gwa_feature_name: Optional[str],
    cdf_quantiles: list[int] | None = None,
) -> Tuple[list[str], np.ndarray]:
    """Build feature columns and finite mask for the hybrid tail model.

    Same as baseline features but adds CDF context columns (e.g. q50, q90
    from each wind source) and returns no monotonic constraint (only 6
    quantile levels, constraint too rigid).

    Returns
    -------
    feat_cols : list[str]
        Ordered feature column names.
    require_finite : np.ndarray
        Boolean mask — True for columns that require finite values.
    """
    if cdf_quantiles is None:
        cdf_quantiles = [50, 90]

    feat_cols: list[str] = ["qnum"]
    feat_cols.extend(wind_cols)

    # CDF context columns
    for wc in wind_cols:
        for q in cdf_quantiles:
            feat_cols.append(f"{wc}_q{q:03d}")

    # Auxiliary features
    feat_cols.extend([c for c in AUX_COLS if c in df.columns])

    # GWA
    if include_gwa and gwa_feature_name and gwa_feature_name in df.columns:
        feat_cols.append(gwa_feature_name)

    # Filter to columns that exist
    feat_cols = [c for c in feat_cols if c in df.columns]

    # Require finite mask: all except GWA
    require_finite = np.ones(len(feat_cols), dtype=bool)
    if include_gwa and gwa_feature_name and gwa_feature_name in feat_cols:
        require_finite[feat_cols.index(gwa_feature_name)] = False

    return feat_cols, require_finite


# ---------------------------------------------------------------------------
# Hybrid fold worker
# ---------------------------------------------------------------------------


def run_one_fold_hybrid(
    sid: str,
    X_base: np.ndarray,
    X_tail: np.ndarray,
    y_full: np.ndarray,
    qnums: np.ndarray,
    station_ids: np.ndarray,
    is_gs: np.ndarray,
    nbr_map: Dict[str, Set[str]],
    base_args_dict: dict,
    tail_xgb_params: dict,
    require_finite_base: np.ndarray,
    require_finite_tail: np.ndarray,
    tail_cutoff: int,
    log_floor: float,
) -> Tuple[str, np.ndarray, np.ndarray, Optional[dict]]:
    """Run one hybrid LOOCV fold: baseline for q<cutoff, tail model for q>=cutoff.

    Parameters
    ----------
    sid : str
        Station ID for the held-out test station.
    X_base, X_tail : np.ndarray
        Feature arrays for base and tail models (same row count).
    y_full : np.ndarray
        (N,) target values.
    qnums : np.ndarray
        (N,) quantile indices.
    station_ids, is_gs : np.ndarray
        Station ID and GS boolean arrays.
    nbr_map : dict
        Neighbor map.
    base_args_dict : dict
        Args dict for ``run_one_fold`` (base model).
    tail_xgb_params : dict
        XGBoost params for the tail model (no monotonic constraint).
    require_finite_base, require_finite_tail : np.ndarray
        Boolean masks over columns.
    tail_cutoff : int
        Quantile index cutoff (e.g. 95). q >= cutoff uses tail model.
    log_floor : float
        Floor for log transform to avoid log(0).

    Returns
    -------
    tuple
        ``(sid, test_idx, combined_pred, metrics_dict_or_None)``
    """
    # --- Base model: run standard fold on ALL quantiles ---
    sid_out, test_idx, base_pred, base_metrics = run_one_fold(
        sid, X_base, y_full, station_ids, is_gs, nbr_map,
        base_args_dict, require_finite_mask=require_finite_base,
    )

    if test_idx.size == 0:
        return sid, test_idx, base_pred, None

    # --- Tail model ---
    seed = base_args_dict["seed"]
    balance_strategy = base_args_dict["balance_strategy"]

    # Replicate exclusion + balancing (same logic as run_one_fold)
    excl: Set[str] = {sid}
    if sid in nbr_map:
        excl |= set(nbr_map[sid])

    base_train_mask = (
        ~np.isin(station_ids, np.fromiter(excl, dtype=station_ids.dtype))
        & np.isfinite(y_full)
    )
    base_idx = np.where(base_train_mask)[0]
    if base_idx.size == 0:
        return sid, test_idx, base_pred, None

    # Require finite on base features (same mask as base model uses)
    req_cols = require_finite_base
    if req_cols.any():
        finite_feat_mask = np.all(np.isfinite(X_base[base_idx][:, req_cols]), axis=1)
    else:
        finite_feat_mask = np.ones(base_idx.shape[0], dtype=bool)

    good_train_idx = base_idx[finite_feat_mask]
    if good_train_idx.size < 20:
        return sid, test_idx, base_pred, None

    # Balance GS/ASOS (identical rng state to base model)
    idx_asos = good_train_idx[~is_gs[good_train_idx]]
    idx_gs = good_train_idx[is_gs[good_train_idx]]

    rng = np.random.default_rng(fold_seed(seed, sid))
    train_idx_bal = balance_indices(idx_asos, idx_gs, rng, strategy=balance_strategy)
    if train_idx_bal.size < 20:
        return sid, test_idx, base_pred, None

    rng.shuffle(train_idx_bal)

    # Filter balanced indices to tail quantiles only
    tail_train_mask = qnums[train_idx_bal] >= tail_cutoff
    tail_train_idx = train_idx_bal[tail_train_mask]

    if tail_train_idx.size < 10:
        # Not enough tail training data — fall back to base predictions
        return sid, test_idx, base_pred, None

    # Also require finite on tail-specific features for training rows
    req_tail = require_finite_tail
    if req_tail.any():
        tail_finite = np.all(np.isfinite(X_tail[tail_train_idx][:, req_tail]), axis=1)
        tail_train_idx = tail_train_idx[tail_finite]
    if tail_train_idx.size < 10:
        return sid, test_idx, base_pred, None

    # Log-transform target
    y_tail_train = np.log(np.maximum(y_full[tail_train_idx], log_floor))

    # Train tail model (no monotonic constraint)
    tail_model = XGBRegressor(**tail_xgb_params)
    tail_model.fit(X_tail[tail_train_idx], y_tail_train)

    # Predict on test station's tail quantiles
    tail_test_mask = qnums[test_idx] >= tail_cutoff
    tail_test_idx = test_idx[tail_test_mask]

    if tail_test_idx.size > 0:
        tail_pred_log = tail_model.predict(X_tail[tail_test_idx])
        tail_pred = np.exp(tail_pred_log).astype(np.float32)
    else:
        tail_pred = np.array([], dtype=np.float32)

    # Assemble: base for q < cutoff, tail for q >= cutoff
    combined_pred = base_pred.copy()
    # Map tail predictions back into combined array
    # tail_test_idx are absolute indices; we need positions within test_idx
    tail_positions = np.where(tail_test_mask)[0]
    if tail_pred.size > 0:
        combined_pred[tail_positions] = tail_pred

    # Compute metrics
    y_test = y_full[test_idx]
    good = np.isfinite(y_test) & np.isfinite(combined_pred)
    metrics = None
    if np.any(good):
        diff = combined_pred[good] - y_test[good]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))

        # Tail-only metrics
        tail_good = tail_test_mask & np.isfinite(y_test) & np.isfinite(combined_pred)
        base_good = (~tail_test_mask) & np.isfinite(y_test) & np.isfinite(combined_pred)
        metrics = {
            "rmse": rmse, "mae": mae,
            "tail_rmse": float(np.sqrt(np.mean((combined_pred[tail_good] - y_test[tail_good])**2))) if np.any(tail_good) else np.nan,
            "base_rmse": float(np.sqrt(np.mean((combined_pred[base_good] - y_test[base_good])**2))) if np.any(base_good) else np.nan,
        }

    return sid, test_idx, combined_pred, metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: Path) -> pd.DataFrame:
    """Load input CSV or Parquet file."""
    if not path.exists():
        raise FileNotFoundError(path)
    log(f"[INFO] Loading: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)
    df.reset_index(drop=True, inplace=True)

    for c in ["station_id", "lat", "lon", "observation", "observation_type", "height_m", "qnum"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    return df


# ---------------------------------------------------------------------------
# Metrics reporting
# ---------------------------------------------------------------------------


def report_metrics(
    all_preds: list[np.ndarray],
    all_truth: list[np.ndarray],
    per_site: list[tuple],
    label: str,
):
    """Report pooled and per-station metrics."""
    if all_preds:
        P = np.concatenate(all_preds)
        T = np.concatenate(all_truth)
        rmse = float(np.sqrt(np.mean((P - T) ** 2)))
        mae = float(np.mean(np.abs(P - T)))
        log(f"[METRIC] {label} LOOCV — RMSE={rmse:.4f} m/s, MAE={mae:.4f} m/s (N={len(T)})")
        if per_site:
            mdf = pd.DataFrame(per_site, columns=["station_id", "rmse", "mae"])
            best5 = mdf.nsmallest(5, "rmse")
            worst5 = mdf.nlargest(5, "rmse")
            log("[METRIC] Best 5 GS stations by RMSE:")
            log(best5.to_string(index=False))
            log("[METRIC] Worst 5 GS stations by RMSE:")
            log(worst5.to_string(index=False))
    else:
        log("[WARN] No GS predictions produced; check inputs.")


def report_hybrid_metrics(
    all_preds: list[np.ndarray],
    all_truth: list[np.ndarray],
    all_qnums: list[np.ndarray],
    per_site: list[tuple],
    tail_cutoff: int,
    label: str,
):
    """Report pooled metrics split by base range and tail range."""
    if not all_preds:
        log("[WARN] No GS predictions produced; check inputs.")
        return

    P = np.concatenate(all_preds)
    T = np.concatenate(all_truth)
    Q = np.concatenate(all_qnums)

    good = np.isfinite(P) & np.isfinite(T)
    rmse = float(np.sqrt(np.mean((P[good] - T[good]) ** 2)))
    mae = float(np.mean(np.abs(P[good] - T[good])))
    log(f"[METRIC] {label} LOOCV — Overall RMSE={rmse:.4f} m/s, MAE={mae:.4f} m/s (N={good.sum()})")

    # Base range
    base_mask = good & (Q < tail_cutoff)
    if np.any(base_mask):
        b_rmse = float(np.sqrt(np.mean((P[base_mask] - T[base_mask]) ** 2)))
        b_mae = float(np.mean(np.abs(P[base_mask] - T[base_mask])))
        log(f"[METRIC] {label} q0-q{tail_cutoff - 1} — RMSE={b_rmse:.4f}, MAE={b_mae:.4f} (N={base_mask.sum()})")

    # Tail range
    tail_mask = good & (Q >= tail_cutoff)
    if np.any(tail_mask):
        t_rmse = float(np.sqrt(np.mean((P[tail_mask] - T[tail_mask]) ** 2)))
        t_mae = float(np.mean(np.abs(P[tail_mask] - T[tail_mask])))
        log(f"[METRIC] {label} q{tail_cutoff}-q100 — RMSE={t_rmse:.4f}, MAE={t_mae:.4f} (N={tail_mask.sum()})")

    if per_site:
        mdf = pd.DataFrame(per_site, columns=["station_id", "rmse", "mae", "tail_rmse", "base_rmse"])
        best5 = mdf.nsmallest(5, "rmse")
        worst5 = mdf.nlargest(5, "rmse")
        log("[METRIC] Best 5 GS stations by RMSE:")
        log(best5[["station_id", "rmse", "mae", "tail_rmse"]].to_string(index=False))
        log("[METRIC] Worst 5 GS stations by RMSE:")
        log(worst5[["station_id", "rmse", "mae", "tail_rmse"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def run_long_experiment(
    exp_type: ExperimentType,
    df: pd.DataFrame,
    wind_cols: list[str],
    nbr_map: Dict[str, Set[str]],
    gs_stations: list[str],
    args,
    include_gwa: bool,
    gwa_feature_name: Optional[str],
):
    """Run a long-format experiment (baseline, enriched, or cdf_context)."""
    enriched = exp_type.name == "enriched"
    cdf_context = exp_type.name == "cdf_context"

    # Track columns added by enrichment so we can strip them before saving
    cols_before = set(df.columns)
    if enriched:
        log(f"[INFO] Enriching with CDF features for {len(wind_cols)} wind columns...")
        df = enrich_with_cdf(df, wind_cols)
    elif cdf_context:
        log(f"[INFO] Enriching with CDF context features (q50/q90) for {len(wind_cols)} wind columns...")
        df = enrich_with_cdf_subset(df, wind_cols)
    enrichment_cols = list(set(df.columns) - cols_before)

    # Build features
    if cdf_context:
        feat_cols, require_finite = build_hybrid_tail_features(
            df, wind_cols, include_gwa, gwa_feature_name,
        )
        # Add monotonic constraint on qnum
        mon = [0] * len(feat_cols)
        if "qnum" in feat_cols:
            mon[feat_cols.index("qnum")] = 1
        mon_str = "(" + ",".join(map(str, mon)) + ")"
    else:
        feat_cols, mon_str, require_finite = build_long_features(
            df, wind_cols, include_gwa, gwa_feature_name, enriched=enriched,
        )
    log(f"[INFO] Feature count: {len(feat_cols)}")

    # Build arrays
    X_full = df[feat_cols].to_numpy(dtype="float32")
    y_full = pd.to_numeric(df["observation"], errors="coerce").to_numpy(dtype="float32")
    is_gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS").to_numpy()
    station_ids = df["station_id"].astype(str).to_numpy()

    df["pred_observation"] = np.nan

    # XGB params
    xgb_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        objective="reg:absoluteerror",
        n_jobs=args.xgb_threads,
        random_state=args.seed,
        tree_method="hist",
        early_stopping_rounds=0,
        monotone_constraints=mon_str,
    )

    args_dict = dict(
        balance_strategy=args.balance_strategy,
        val_frac=0.0,
        seed=args.seed,
        xgb_params=xgb_params,
    )

    log(
        f"[INFO] Running {exp_type.name} LOOCV over {len(gs_stations)} GS stations "
        f"with n_jobs={args.n_jobs}"
    )

    results = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="processes")(
        delayed(run_one_fold)(
            sid, X_full, y_full, station_ids, is_gs, nbr_map,
            args_dict, require_finite_mask=require_finite,
        )
        for sid in gs_stations
    )

    # Collect
    all_preds, all_truth, per_site = [], [], []
    for sid, test_idx, pred, metrics in results:
        if test_idx.size > 0 and pred.size == test_idx.size:
            df.loc[test_idx, "pred_observation"] = pred
            yt = y_full[test_idx]
            good = np.isfinite(yt) & np.isfinite(pred)
            if np.any(good):
                all_truth.append(yt[good])
                all_preds.append(pred[good])
                if metrics:
                    per_site.append((sid, metrics[0], metrics[1]))

    report_metrics(all_preds, all_truth, per_site, exp_type.name.capitalize())

    # Strip enrichment columns before saving
    output_df = df.drop(columns=enrichment_cols) if enrichment_cols else df

    # Save
    outfile = args.outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    log(f"[INFO] Saving predictions → {outfile}")
    if outfile.suffix.lower() == ".parquet":
        output_df.to_parquet(outfile, index=False)
    else:
        output_df.to_csv(outfile, index=False)


def run_wide_experiment(
    exp_type: ExperimentType,
    df: pd.DataFrame,
    wind_cols: list[str],
    nbr_map: Dict[str, Set[str]],
    gs_stations: list[str],
    args,
    include_gwa: bool,
    gwa_feature_name: Optional[str],
):
    """Run a wide-format experiment."""
    log(f"[INFO] Pivoting to wide format with wind features: {wind_cols}")
    wide_df = pivot_to_wide(df, wind_cols, target_col="observation")
    log(f"[INFO] Wide table: {wide_df.shape[0]} rows x {wide_df.shape[1]} cols")

    feat_cols, obs_cols = build_wide_features(
        wide_df, wind_cols, include_gwa, gwa_feature_name,
    )
    log(f"[INFO] Feature count: {len(feat_cols)}")

    missing_obs = [c for c in obs_cols if c not in wide_df.columns]
    if missing_obs:
        raise ValueError(f"Missing target columns after pivot: {missing_obs[:5]}...")

    X_wide = wide_df[feat_cols].to_numpy(dtype="float32")
    Y_wide = wide_df[obs_cols].to_numpy(dtype="float32")

    station_ids = wide_df["station_id"].astype(str).to_numpy()
    is_gs = (
        wide_df["observation_type"].astype(str).map(normalize_obs_type).eq("GS").to_numpy()
    )

    # Filter GS stations to those present in wide_df
    wide_gs = set(pd.Series(station_ids[is_gs]).unique())
    gs_stations = [s for s in gs_stations if s in wide_gs]
    if not gs_stations:
        raise SystemExit("No GS stations found in wide-format data")
    log(f"[INFO] GS stations for LOOCV: {len(gs_stations)}")

    xgb_params = {
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "objective": "reg:absoluteerror",
        "tree_method": "hist",
        "n_jobs": args.xgb_threads,
        "random_state": args.seed,
    }

    log(
        f"[INFO] Running wide LOOCV over {len(gs_stations)} GS stations "
        f"with n_jobs={args.n_jobs}"
    )

    results = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="processes")(
        delayed(run_one_fold_wide)(
            sid, X_wide, Y_wide, station_ids, is_gs, nbr_map,
            xgb_params, args.seed, args.balance_strategy,
        )
        for sid in gs_stations
    )

    # Collect
    preds_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    all_preds, all_truth, per_site = [], [], []

    for sid, test_idx, preds, metrics in results:
        if test_idx.size == 0:
            continue
        preds_dict[sid] = (test_idx, preds)
        Y_test = Y_wide[test_idx]
        good = np.isfinite(Y_test) & np.isfinite(preds)
        if np.any(good):
            all_truth.append(Y_test[good])
            all_preds.append(preds[good])
            if metrics:
                per_site.append((sid, metrics[0], metrics[1]))

    long_df = wide_preds_to_long(wide_df, preds_dict)
    log(f"[INFO] Long output: {len(long_df)} rows")

    report_metrics(all_preds, all_truth, per_site, "Wide")

    # Save
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.outfile, index=False)
    log(f"[INFO] Saved -> {args.outfile}")


def run_convnet_experiment(
    exp_type: ExperimentType,
    df: pd.DataFrame,
    wind_cols: list[str],
    nbr_map: Dict[str, Set[str]],
    gs_stations: list[str],
    args,
    include_gwa: bool,
    gwa_feature_name: Optional[str],
):
    """Run a ConvNet CDF-to-CDF experiment."""
    from wem.experiment.convnet import run_one_fold_convnet

    log(f"[INFO] Pivoting to wide format with wind features: {wind_cols}")
    wide_df = pivot_to_wide(df, wind_cols, target_col="observation")
    log(f"[INFO] Wide table: {wide_df.shape[0]} rows x {wide_df.shape[1]} cols")

    cdf_input, aux_input, targets = wide_to_convnet_arrays(
        wide_df, wind_cols, include_gwa, gwa_feature_name,
    )
    log(f"[INFO] ConvNet arrays — CDF: {cdf_input.shape}, aux: {aux_input.shape}, target: {targets.shape}")

    station_ids = wide_df["station_id"].astype(str).to_numpy()
    is_gs = (
        wide_df["observation_type"].astype(str).map(normalize_obs_type).eq("GS").to_numpy()
    )

    # Filter GS stations to those present in wide_df
    wide_gs = set(pd.Series(station_ids[is_gs]).unique())
    gs_stations = [s for s in gs_stations if s in wide_gs]
    if not gs_stations:
        raise SystemExit("No GS stations found in wide-format data")
    log(f"[INFO] GS stations for LOOCV: {len(gs_stations)}")

    # Sequential LOOCV (GPU-bound, no joblib parallelism)
    preds_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    all_preds, all_truth, per_site = [], [], []

    for i, sid in enumerate(gs_stations):
        log(f"[INFO] ConvNet fold {i + 1}/{len(gs_stations)}: {sid}")
        sid_out, test_idx, preds, metrics = run_one_fold_convnet(
            sid,
            cdf_input,
            aux_input,
            targets,
            station_ids,
            is_gs,
            nbr_map,
            seed=args.seed,
            balance_strategy=args.balance_strategy,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
            val_frac=args.val_frac,
            device=args.device,
            n_conv_layers=args.n_conv_layers,
        )

        if test_idx.size == 0:
            continue
        preds_dict[sid] = (test_idx, preds)
        Y_test = targets[test_idx]
        good = np.isfinite(Y_test) & np.isfinite(preds)
        if np.any(good):
            all_truth.append(Y_test[good])
            all_preds.append(preds[good])
            if metrics:
                per_site.append((sid, metrics[0], metrics[1]))

    long_df = wide_preds_to_long(wide_df, preds_dict)
    log(f"[INFO] Long output: {len(long_df)} rows")

    report_metrics(all_preds, all_truth, per_site, "ConvNet")

    # Save
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.outfile, index=False)
    log(f"[INFO] Saved -> {args.outfile}")


def run_mlp_experiment(
    exp_type: ExperimentType,
    df: pd.DataFrame,
    wind_cols: list[str],
    nbr_map: Dict[str, Set[str]],
    gs_stations: list[str],
    args,
    include_gwa: bool,
    gwa_feature_name: Optional[str],
):
    """Run a long-format MLP experiment (same features as baseline)."""
    from wem.experiment.mlp import run_one_fold_mlp

    # Build features — identical to baseline long format
    feat_cols, _mon_str, require_finite = build_long_features(
        df, wind_cols, include_gwa, gwa_feature_name, enriched=False,
    )
    log(f"[INFO] Feature count: {len(feat_cols)}")

    # Build arrays
    X_full = df[feat_cols].to_numpy(dtype="float32")
    y_full = pd.to_numeric(df["observation"], errors="coerce").to_numpy(dtype="float32")
    is_gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS").to_numpy()
    station_ids = df["station_id"].astype(str).to_numpy()

    df["pred_observation"] = np.nan

    # Parse hidden dims
    hidden_dims = tuple(args.hidden_dims)

    # Sequential LOOCV (GPU-bound, no joblib parallelism)
    all_preds, all_truth, per_site = [], [], []

    for i, sid in enumerate(gs_stations):
        log(f"[INFO] MLP fold {i + 1}/{len(gs_stations)}: {sid}")
        sid_out, test_idx, pred, metrics = run_one_fold_mlp(
            sid,
            X_full,
            y_full,
            station_ids,
            is_gs,
            nbr_map,
            seed=args.seed,
            balance_strategy=args.balance_strategy,
            require_finite_mask=require_finite,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
            val_frac=args.val_frac,
            device=args.device,
            hidden_dims=hidden_dims,
        )

        if test_idx.size > 0 and pred.size == test_idx.size:
            df.loc[test_idx, "pred_observation"] = pred
            yt = y_full[test_idx]
            good = np.isfinite(yt) & np.isfinite(pred)
            if np.any(good):
                all_truth.append(yt[good])
                all_preds.append(pred[good])
                if metrics:
                    per_site.append((sid, metrics[0], metrics[1]))

    report_metrics(all_preds, all_truth, per_site, "MLP")

    # Save
    outfile = args.outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    log(f"[INFO] Saving predictions → {outfile}")
    if outfile.suffix.lower() == ".parquet":
        df.to_parquet(outfile, index=False)
    else:
        df.to_csv(outfile, index=False)


def run_hybrid_experiment(
    exp_type: ExperimentType,
    df: pd.DataFrame,
    wind_cols: list[str],
    nbr_map: Dict[str, Set[str]],
    gs_stations: list[str],
    args,
    include_gwa: bool,
    gwa_feature_name: Optional[str],
):
    """Run a hybrid experiment: baseline q0-q94, log-target tail model q95-q100."""
    tail_cutoff = args.tail_cutoff
    log_floor = args.tail_log_floor
    use_cdf_context = args.tail_cdf_context

    # Build base features (identical to baseline)
    enrichment_cols: list[str] = []
    if use_cdf_context:
        cols_before = set(df.columns)
        log("[INFO] Enriching with CDF subset features for tail model...")
        df = enrich_with_cdf_subset(df, wind_cols)
        enrichment_cols = list(set(df.columns) - cols_before)

    base_feat_cols, base_mon_str, require_finite_base = build_long_features(
        df, wind_cols, include_gwa, gwa_feature_name, enriched=False,
    )
    log(f"[INFO] Base feature count: {len(base_feat_cols)}")

    # Build tail features
    if use_cdf_context:
        tail_feat_cols, require_finite_tail = build_hybrid_tail_features(
            df, wind_cols, include_gwa, gwa_feature_name,
        )
    else:
        # Same features as base, just no monotonic constraint + log target
        tail_feat_cols = base_feat_cols
        require_finite_tail = require_finite_base.copy()
    log(f"[INFO] Tail feature count: {len(tail_feat_cols)}")

    # Build arrays
    X_base = df[base_feat_cols].to_numpy(dtype="float32")
    X_tail = df[tail_feat_cols].to_numpy(dtype="float32")
    y_full = pd.to_numeric(df["observation"], errors="coerce").to_numpy(dtype="float32")
    is_gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS").to_numpy()
    station_ids = df["station_id"].astype(str).to_numpy()
    qnums = df["qnum"].to_numpy(dtype=int)

    df["pred_observation"] = np.nan

    # Base model XGB params (with monotonic constraint)
    base_xgb_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        objective="reg:absoluteerror",
        n_jobs=args.xgb_threads,
        random_state=args.seed,
        tree_method="hist",
        early_stopping_rounds=0,
        monotone_constraints=base_mon_str,
    )

    base_args_dict = dict(
        balance_strategy=args.balance_strategy,
        val_frac=0.0,
        seed=args.seed,
        xgb_params=base_xgb_params,
    )

    # Tail model XGB params (no monotonic constraint)
    tail_xgb_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        objective="reg:absoluteerror",
        n_jobs=args.xgb_threads,
        random_state=args.seed,
        tree_method="hist",
    )

    log(
        f"[INFO] Running hybrid LOOCV over {len(gs_stations)} GS stations "
        f"(tail cutoff q>={tail_cutoff}) with n_jobs={args.n_jobs}"
    )

    results = Parallel(n_jobs=args.n_jobs, verbose=10, prefer="processes")(
        delayed(run_one_fold_hybrid)(
            sid, X_base, X_tail, y_full, qnums, station_ids, is_gs, nbr_map,
            base_args_dict, tail_xgb_params, require_finite_base,
            require_finite_tail, tail_cutoff, log_floor,
        )
        for sid in gs_stations
    )

    # Collect
    all_preds, all_truth, all_qnums_list, per_site = [], [], [], []
    for sid, test_idx, pred, metrics in results:
        if test_idx.size > 0 and pred.size == test_idx.size:
            df.loc[test_idx, "pred_observation"] = pred
            yt = y_full[test_idx]
            qt = qnums[test_idx]
            good = np.isfinite(yt) & np.isfinite(pred)
            if np.any(good):
                all_truth.append(yt[good])
                all_preds.append(pred[good])
                all_qnums_list.append(qt[good])
                if metrics:
                    per_site.append((
                        sid, metrics["rmse"], metrics["mae"],
                        metrics["tail_rmse"], metrics["base_rmse"],
                    ))

    report_hybrid_metrics(
        all_preds, all_truth, all_qnums_list, per_site, tail_cutoff, "Hybrid",
    )

    # Strip enrichment columns before saving
    output_df = df.drop(columns=enrichment_cols) if enrichment_cols else df

    # Save
    outfile = args.outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    log(f"[INFO] Saving predictions → {outfile}")
    if outfile.suffix.lower() == ".parquet":
        output_df.to_parquet(outfile, index=False)
    else:
        output_df.to_csv(outfile, index=False)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_experiment(exp_type: ExperimentType, args):
    """Load data, run experiment, save results."""
    # Check output
    if args.outfile.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.outfile}  (use --overwrite)")

    # Load
    df = load_data(args.infile)

    # Prepare features & neighbors
    df = make_features(df)
    nbr_map = build_neighbor_map(df)

    # Optional GWA merge
    gwa_feature_name: Optional[str] = None
    if args.gwa_file is not None:
        df, gwa_feature_name = merge_gwa_feature(df, args.gwa_file, gwa_col="gwa_interp")

    # Parse wind features
    valid_wind = set(WIND_FEATURE_MAP)
    sel_wind = [s.strip() for s in args.wind_features.split(",") if s.strip()]
    bad = [w for w in sel_wind if w not in valid_wind]
    if bad:
        raise ValueError(f"Invalid --wind-features: {bad} (valid: {sorted(valid_wind)})")
    wind_cols = [WIND_FEATURE_MAP[w] for w in sel_wind]

    # Determine GS stations for LOOCV
    is_gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS")
    station_ids_series = df["station_id"].astype(str)
    gs_stations = station_ids_series[is_gs].unique().tolist()

    if args.stations:
        subset = [s.strip() for s in args.stations.split(",") if s.strip()]
        gs_stations = [s for s in gs_stations if s in subset]
        if not gs_stations:
            raise SystemExit("None of --stations found in GS stations")
    log(f"[INFO] GS stations for LOOCV: {len(gs_stations)}")

    include_gwa = args.include_gwa and gwa_feature_name is not None

    if exp_type.format == "long":
        run_long_experiment(
            exp_type, df, wind_cols, nbr_map, gs_stations, args,
            include_gwa, gwa_feature_name,
        )
    elif exp_type.format == "wide":
        run_wide_experiment(
            exp_type, df, wind_cols, nbr_map, gs_stations, args,
            include_gwa, gwa_feature_name,
        )
    elif exp_type.format == "convnet":
        run_convnet_experiment(
            exp_type, df, wind_cols, nbr_map, gs_stations, args,
            include_gwa, gwa_feature_name,
        )
    elif exp_type.format == "mlp":
        run_mlp_experiment(
            exp_type, df, wind_cols, nbr_map, gs_stations, args,
            include_gwa, gwa_feature_name,
        )
    elif exp_type.format == "hybrid":
        run_hybrid_experiment(
            exp_type, df, wind_cols, nbr_map, gs_stations, args,
            include_gwa, gwa_feature_name,
        )

    log("[INFO] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared across all experiment subcommands."""
    parser.add_argument(
        "--infile", type=Path,
        default=Path("combined_quantiles_long_with_topo_loocv_10km.csv"),
        help="Input long-format CSV (or Parquet).",
    )
    parser.add_argument(
        "--outfile", type=Path, required=True,
        help="Output CSV with predictions.",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output if it exists.")

    # Feature controls
    parser.add_argument(
        "--wind-features", type=str, default="hrrr,wtk,wtk_led_conus",
        help="Comma-separated wind resource columns.",
    )

    # GWA
    parser.add_argument("--gwa-file", type=Path, default=None,
                        help="GWA CSV for merge.")
    parser.add_argument("--include-gwa", action="store_true",
                        help="Include GWA as feature.")

    # Hyperparameters (defaults from production Optuna tuning)
    parser.add_argument("--learning-rate", type=float,
                        default=DEFAULT_XGB_PARAMS["learning_rate"])
    parser.add_argument("--max-depth", type=int,
                        default=DEFAULT_XGB_PARAMS["max_depth"])
    parser.add_argument("--n-estimators", type=int,
                        default=DEFAULT_XGB_PARAMS["n_estimators"])
    parser.add_argument("--subsample", type=float,
                        default=DEFAULT_XGB_PARAMS["subsample"])
    parser.add_argument("--colsample-bytree", type=float,
                        default=DEFAULT_XGB_PARAMS["colsample_bytree"])
    parser.add_argument("--min-child-weight", type=float,
                        default=DEFAULT_XGB_PARAMS["min_child_weight"])

    # Execution
    parser.add_argument("--balance-strategy", type=str, default="downsample",
                        choices=["downsample", "upsample", "none"])
    parser.add_argument("--n-jobs", type=int, default=12,
                        help="Parallel LOOCV folds.")
    parser.add_argument("--xgb-threads", type=int, default=1,
                        help="Threads per XGBoost model (default 1). "
                             "Increase for high-feature experiments to reduce IPC overhead.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stations", type=str, default=None,
                        help="Comma-separated GS station IDs (subset).")


def _add_convnet_args(parser: argparse.ArgumentParser):
    """Add ConvNet-specific arguments to a subparser."""
    parser.add_argument("--epochs", type=int, default=300,
                        help="Max training epochs (default 300).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default 32).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default 1e-4).")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default 0.3).")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience in epochs (default 30).")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of training stations for validation (default 0.2).")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="PyTorch device (default cpu).")
    parser.add_argument("--n-conv-layers", type=int, default=3,
                        choices=[2, 3],
                        help="Number of conv blocks (default 3).")


def _add_mlp_args(parser: argparse.ArgumentParser):
    """Add MLP-specific arguments to a subparser."""
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs (default 100).")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size (default 512).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default 1e-4).")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default 0.3).")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience in epochs (default 15).")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of training stations for validation (default 0.2).")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="PyTorch device (default cpu).")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64],
                        help="Hidden layer sizes (default: 128 64).")


def _add_hybrid_args(parser: argparse.ArgumentParser):
    """Add hybrid-specific arguments to a subparser."""
    parser.add_argument("--tail-cutoff", type=int, default=95,
                        help="Quantile index cutoff for tail model (default 95).")
    parser.add_argument("--tail-log-floor", type=float, default=1e-6,
                        help="Floor for log transform of target (default 1e-6).")
    parser.add_argument("--tail-cdf-context", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Add CDF context features (q50/q90) to tail model (default: yes). "
                             "Use --no-tail-cdf-context for identical features to baseline.")


def main():
    ap = argparse.ArgumentParser(
        description="WEM experiment runner — unified CLI for LOOCV experiments.",
    )
    sub = ap.add_subparsers(dest="command")

    # Experiment subcommands
    for name, exp in EXPERIMENTS.items():
        sp = sub.add_parser(name, help=exp.description)
        _add_common_args(sp)
        if exp.format == "convnet":
            _add_convnet_args(sp)
        elif exp.format == "mlp":
            _add_mlp_args(sp)
        elif exp.format == "hybrid":
            _add_hybrid_args(sp)

    # Compare subcommand
    cmp = sub.add_parser("compare", help="Compare experiment results")
    cmp.add_argument("--baseline", type=Path, required=True,
                     help="Baseline predictions CSV.")
    cmp.add_argument("--experiments", type=Path, nargs="+", required=True,
                     help="One or more experiment prediction CSVs.")
    cmp.add_argument("--labels", type=str, nargs="*", default=None,
                     help="Display labels for experiments (default: from filenames).")
    cmp.add_argument("--hybrid-cutoff", type=int, default=None, metavar="Q",
                     help="Hybrid: experiment for q < Q, baseline for q >= Q.")
    cmp.add_argument("--top-n", type=int, default=0,
                     help="Show only top/bottom N stations.")
    cmp.add_argument("--save-csv", type=Path, default=None,
                     help="Save per-station comparison to CSV.")
    cmp.add_argument("--quantile-detail", action="store_true",
                     help="Print full per-quantile table.")

    args = ap.parse_args()

    if args.command is None:
        ap.print_help()
        raise SystemExit(1)

    if args.command == "compare":
        from wem.experiment.compare import run_compare
        run_compare(args)
    elif args.command in EXPERIMENTS:
        run_experiment(EXPERIMENTS[args.command], args)
    else:
        ap.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
