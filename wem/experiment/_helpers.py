"""Shared pure functions for experiment infrastructure modules.

Consolidates duplicated code from the original dev scripts:
optimize_hyperparams, optimize_n_estimators, optimize_max_depth,
sweep_wind_features, sweep_aux_features, analyze_feature_sweeps,
analyze_aux_sweeps.
"""

from __future__ import annotations

import shlex
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from wem.utils.sites import normalize_obs_type


# ---------- basic metrics ----------

def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root-mean-square error."""
    e = yhat - y
    return float(np.sqrt(np.mean(e * e)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(yhat - y)))


def ci95(x: np.ndarray) -> tuple[float, float]:
    """Return (lo, hi) for a 95% confidence interval of the mean.

    Returns (nan, nan) when fewer than 2 observations.
    """
    n = x.size
    if n < 2:
        return (np.nan, np.nan)
    mu = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    return (mu - 1.96 * se, mu + 1.96 * se)


# ---------- CSV-level metrics ----------

def compute_gs_metrics(csv_path: Path) -> tuple[float, float, int]:
    """Compute (rmse, mae, n) on GS rows between observation and pred_observation.

    Returns (inf, inf, 0) if the file is missing, unreadable, or has no valid data.
    """
    if not csv_path.exists():
        return float("inf"), float("inf"), 0
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return float("inf"), float("inf"), 0

    required = {"observation", "pred_observation", "observation_type"}
    if not required.issubset(df.columns):
        return float("inf"), float("inf"), 0

    gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS")
    sub = df.loc[gs, ["observation", "pred_observation"]].copy()
    sub["observation"] = pd.to_numeric(sub["observation"], errors="coerce")
    sub["pred_observation"] = pd.to_numeric(sub["pred_observation"], errors="coerce")
    sub = sub[np.isfinite(sub["observation"]) & np.isfinite(sub["pred_observation"])]
    if sub.empty:
        return float("inf"), float("inf"), 0

    y = sub["observation"].to_numpy(dtype=float)
    yhat = sub["pred_observation"].to_numpy(dtype=float)
    r = float(np.sqrt(np.mean((yhat - y) ** 2)))
    m = float(np.mean(np.abs(yhat - y)))
    return r, m, int(len(sub))


def compute_site_mae(csv_path: Path) -> tuple[pd.DataFrame, float, float, int]:
    """Compute per-site MAE on GS rows.

    Returns (per_site_df, mean_mae, median_mae, n_sites).
    per_site_df columns: ['station_id', 'mae'].
    Returns empty df and (inf, inf, 0) on failure.
    """
    empty = pd.DataFrame(columns=["station_id", "mae"])
    if not csv_path.exists():
        return empty, float("inf"), float("inf"), 0
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return empty, float("inf"), float("inf"), 0

    req = {"station_id", "observation_type", "observation", "pred_observation"}
    if not req.issubset(df.columns):
        return empty, float("inf"), float("inf"), 0

    gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS")
    sub = df.loc[gs, ["station_id", "observation", "pred_observation"]].copy()
    sub["observation"] = pd.to_numeric(sub["observation"], errors="coerce")
    sub["pred_observation"] = pd.to_numeric(sub["pred_observation"], errors="coerce")
    sub = sub[np.isfinite(sub["observation"]) & np.isfinite(sub["pred_observation"])]

    if sub.empty:
        return empty, float("inf"), float("inf"), 0

    site_mae = (
        sub.assign(abs_err=lambda d: np.abs(d["pred_observation"] - d["observation"]))
        .groupby("station_id", as_index=False)["abs_err"]
        .mean()
        .rename(columns={"abs_err": "mae"})
    )

    if site_mae.empty:
        return empty, float("inf"), float("inf"), 0

    mean_mae = float(site_mae["mae"].mean())
    median_mae = float(site_mae["mae"].median())
    return site_mae[["station_id", "mae"]], mean_mae, median_mae, int(site_mae.shape[0])


def compute_sweep_metrics(
    df: pd.DataFrame, stationwise: bool = False
) -> dict[str, float]:
    """Compute RMSE/MAE on GS rows, optionally station-averaged.

    Returns dict with keys: rmse, mae, rows, stations, rmse_stationwise,
    mae_stationwise.
    """
    gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS")
    sub = df.loc[gs, ["station_id", "observation", "pred_observation"]].copy()
    sub["observation"] = pd.to_numeric(sub["observation"], errors="coerce")
    sub["pred_observation"] = pd.to_numeric(sub["pred_observation"], errors="coerce")
    sub = sub[np.isfinite(sub["observation"]) & np.isfinite(sub["pred_observation"])]
    if sub.empty:
        return {
            "rmse": np.nan, "mae": np.nan, "rows": 0, "stations": 0,
            "rmse_stationwise": np.nan, "mae_stationwise": np.nan,
        }

    if stationwise:
        stats = []
        for _, grp in sub.groupby("station_id"):
            y = grp["observation"].to_numpy(dtype=float)
            yhat = grp["pred_observation"].to_numpy(dtype=float)
            if y.size > 0:
                stats.append((rmse(y, yhat), mae(y, yhat)))
        r_all = float(np.mean([s[0] for s in stats])) if stats else np.nan
        m_all = float(np.mean([s[1] for s in stats])) if stats else np.nan
    else:
        y = sub["observation"].to_numpy(dtype=float)
        yhat = sub["pred_observation"].to_numpy(dtype=float)
        r_all = rmse(y, yhat)
        m_all = mae(y, yhat)

    # Always compute stationwise for reference
    stats = []
    for _, grp in sub.groupby("station_id"):
        y = grp["observation"].to_numpy(dtype=float)
        yhat = grp["pred_observation"].to_numpy(dtype=float)
        if y.size > 0:
            stats.append((rmse(y, yhat), mae(y, yhat)))
    r_sw = float(np.mean([s[0] for s in stats])) if stats else np.nan
    m_sw = float(np.mean([s[1] for s in stats])) if stats else np.nan

    return {
        "rmse": r_all, "mae": m_all,
        "rows": int(len(sub)), "stations": int(sub["station_id"].nunique()),
        "rmse_stationwise": r_sw, "mae_stationwise": m_sw,
    }


# ---------- label / path parsing ----------

def parse_label_from_path(p: Path, prefix: str = "ml_results_xgb_") -> str:
    """Extract the feature-combo label from a sweep output filename.

    Given ``ml_results_xgb_era5+wtk.csv`` with the default prefix,
    returns ``'era5+wtk'``.
    """
    stem = p.stem
    if stem.startswith(prefix):
        return stem[len(prefix):] or "none"
    return stem


def features_from_label(label: str) -> tuple[str, tuple[str, ...]]:
    """Parse a '+'-separated label into (label, tuple_of_features).

    ``'none'`` or ``''`` returns an empty tuple.
    """
    if label in ("none", "", "(none)"):
        return label, tuple()
    feats = tuple(x for x in label.split("+") if x)
    return label, feats


# ---------- marginal delta analysis ----------

def make_pairs_for_feature(
    all_runs: dict[tuple[str, ...], dict[str, float]],
    feature: str,
    key_metric: str = "rmse",
) -> list[tuple[tuple[str, ...], float]]:
    """Build paired comparisons for a feature.

    For each base set S (not containing *feature*), computes
    delta = metric(S | {feature}) - metric(S).
    Returns list of (S, delta) where both runs exist and are finite.
    """
    deltas: list[tuple[tuple[str, ...], float]] = []
    for S in list(all_runs.keys()):
        if feature in S:
            continue
        S_with = tuple(sorted(S + (feature,)))
        if S in all_runs and S_with in all_runs:
            a = all_runs[S_with].get(key_metric, np.nan)
            b = all_runs[S].get(key_metric, np.nan)
            if np.isfinite(a) and np.isfinite(b):
                deltas.append((S, float(a - b)))
    return deltas


# ---------- feature enumeration ----------

def combo_label(feats: list[str]) -> str:
    """Return a '+'-joined label for a feature combination ('none' if empty)."""
    if not feats:
        return "none"
    return "+".join(feats)


def enumerate_subsets(
    items: list[str], include_empty: bool = False
) -> list[list[str]]:
    """Return all subsets of *items*, preserving original order.

    If *include_empty* is False, the empty set is excluded.
    """
    start = 0 if include_empty else 1
    subsets: list[list[str]] = []
    for r in range(start, len(items) + 1):
        for c in combinations(items, r):
            subsets.append(list(c))
    return subsets


# ---------- training command construction ----------

def resolve_train_command(train_cmd: str | None = None) -> list[str]:
    """Return a list of command tokens to invoke the LOOCV training script.

    If *train_cmd* is provided, it is split with ``shlex.split``.
    Otherwise, returns ``[sys.executable, '-m', 'wem.train.loocv_xgb']``.
    """
    if train_cmd:
        return shlex.split(train_cmd)
    return [sys.executable, "-m", "wem.train.loocv_xgb"]


def build_train_cmd(
    train_cmd: list[str],
    infile: Path,
    outfile: Path,
    *,
    params: dict[str, object] | None = None,
    wind_features: str | None = None,
    aux_features: str | None = None,
    balance_strategy: str = "downsample",
    gs_only: bool = False,
    n_jobs_outer: int | None = None,
    n_jobs_model: int | None = None,
    seed: int = 42,
    extra_args: str = "",
) -> list[str]:
    """Build a full argument list for invoking the LOOCV training script.

    Always appends ``--val_frac 0.0 --early_stopping_rounds 0``.
    *params* is a dict of XGBoost hyperparameters (e.g. learning_rate, max_depth).
    """
    cmd = list(train_cmd) + [
        "--infile", str(infile),
        "--outfile", str(outfile),
    ]

    if params:
        for k, v in params.items():
            if isinstance(v, float):
                cmd.extend([f"--{k}", f"{v:.6f}"])
            else:
                cmd.extend([f"--{k}", str(v)])

    cmd.extend(["--balance_strategy", balance_strategy])

    if gs_only:
        cmd.append("--gs_only")

    if wind_features is not None:
        cmd.extend(["--wind_features", wind_features])
    if aux_features is not None:
        cmd.extend(["--aux_features", aux_features])

    if n_jobs_outer is not None:
        cmd.extend(["--n_jobs_outer", str(n_jobs_outer)])
    if n_jobs_model is not None:
        cmd.extend(["--n_jobs_model", str(n_jobs_model)])

    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--val_frac", "0.0", "--early_stopping_rounds", "0"])

    if extra_args and extra_args.strip():
        cmd.extend(shlex.split(extra_args))

    return cmd
