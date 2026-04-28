"""Hyperparameter optimization (serial) for LOOCV XGBoost using Optuna.

Runs trials one-by-one (no parallel trials). Each trial calls the training
script, overriding key XGB hyperparameters via CLI. Metric is computed from
the produced CSV (GS-only MAE by default). The Optuna study is persisted to
disk (SQLite) so runs can be resumed.

Example::

    wem-exp-hpo \\
        --infile combined_quantiles_long_with_topo_loocv_10km.csv \\
        --results_dir results_hpo_xgb \\
        --n_trials 40 \\
        --wind_features hrrr,wtk,wtk_led_conus \\
        --aux_features latlon,height,elevation \\
        --gs_only \\
        --balance_strategy upsample
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as e:
    raise SystemExit(
        "This script requires Optuna. Install with `pip install optuna`."
    ) from e

from wem.experiment._helpers import (
    build_train_cmd,
    compute_gs_metrics,
    resolve_train_command,
)
from wem.utils.logging import log


def normalize_storage_arg(results_dir: Path, storage: str | None) -> str:
    """Return a proper Optuna storage URL.

    Defaults to a sqlite file under results_dir/optuna_study.db.
    """
    if not storage or str(storage).strip().lower() in {"", "default"}:
        db_path = (results_dir / "optuna_study.db").resolve()
        return f"sqlite:///{db_path}"
    s = str(storage)
    if "://" in s:
        return s
    db_path = Path(s).expanduser().resolve()
    return f"sqlite:///{db_path}"


def make_objective(args) -> "optuna.trial.Trial":  # noqa: F821
    """Create an Optuna objective function closed over *args*."""
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    train_cmd = resolve_train_command(args.train_cmd)

    def objective(trial: optuna.trial.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        max_depth = trial.suggest_int("max_depth", 6, 20)
        min_child_weight = trial.suggest_float("min_child_weight", 0.2, 10.0, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        n_estimators = trial.suggest_int("n_estimators", 200, 1000, step=20)

        trial_tag = f"{stamp}_t{trial.number:03d}"
        outfile = results_dir / f"ml_results_xgb_hpo_{trial_tag}.csv"
        logfile = results_dir / f"ml_results_xgb_hpo_{trial_tag}.log"

        need_run = not outfile.exists()

        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "n_estimators": n_estimators,
        }

        cmd = build_train_cmd(
            train_cmd,
            args.infile,
            outfile,
            params=params,
            wind_features=args.wind_features,
            aux_features=args.aux_features,
            balance_strategy=args.balance_strategy,
            gs_only=args.gs_only,
            n_jobs_outer=args.n_jobs_outer,
            n_jobs_model=args.n_jobs_model,
            seed=args.seed,
            extra_args=args.extra_args,
        )

        if need_run:
            with open(logfile, "w") as lf:
                lf.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
                lf.flush()
                try:
                    start = time.time()
                    proc = subprocess.run(cmd, stdout=lf, stderr=lf, check=False)
                    dur = time.time() - start
                    lf.write(f"\n[DONE] returncode={proc.returncode} duration_sec={dur:.2f}\n")
                except Exception as e:
                    lf.write(f"\n[ERROR] Exception while running trial: {e}\n")
                    return float("inf")

        rmse_val, mae_val, n = compute_gs_metrics(outfile)

        row = {
            "trial": trial.number,
            "outfile": str(outfile),
            "rmse": rmse_val,
            "mae": mae_val,
            "n_rows_gs": n,
            **params,
            "wind_features": args.wind_features,
            "aux_features": args.aux_features,
            "balance_strategy": args.balance_strategy,
            "gs_only": bool(args.gs_only),
            "seed": args.seed,
        }
        man_path = results_dir / "hpo_manifest.csv"
        if not man_path.exists():
            pd.DataFrame([row]).to_csv(man_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(man_path, index=False, mode="a", header=False)

        return float(mae_val if np.isfinite(mae_val) else 1e9)

    return objective


def main():
    pa = argparse.ArgumentParser(
        description="Serial Optuna HPO for LOOCV XGB training script."
    )
    pa.add_argument(
        "--train-cmd", type=str, default=None,
        help="Training command override. Default: python -m wem.train.loocv_xgb",
    )
    pa.add_argument(
        "--infile", type=Path,
        default=Path("combined_quantiles_long_with_topo_loocv_10km.csv"),
        help="Training data CSV/Parquet for the training script.",
    )
    pa.add_argument(
        "--results_dir", type=Path, default=Path("results_hpo_xgb_upsample"),
        help="Directory to store per-trial outputs and logs.",
    )
    pa.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials.")
    pa.add_argument("--seed", type=int, default=42, help="Random seed.")

    pa.add_argument("--wind_features", type=str, default="hrrr,wtk,wtk_led_conus")
    pa.add_argument("--aux_features", type=str, default="latlon,height,elevation")
    pa.add_argument(
        "--balance_strategy", type=str, default="downsample",
        choices=["downsample", "upsample"],
    )
    pa.add_argument("--gs_only", action="store_true")
    pa.add_argument("--n_jobs_outer", type=int, default=12)
    pa.add_argument("--n_jobs_model", type=int, default=1)

    pa.add_argument(
        "--storage", type=str, default="",
        help="Optuna storage URL or sqlite path. Default: <results_dir>/optuna_study.db",
    )
    pa.add_argument("--study_name", type=str, default="xgb_hpo_v3")
    pa.add_argument("--extra_args", type=str, default="")
    args = pa.parse_args()

    log("[INFO] Running serial Optuna HPO (no parallel trials).")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    storage_url = normalize_storage_arg(args.results_dir, args.storage)
    log(f"[INFO] Optuna storage: {storage_url}")
    log(f"[INFO] Optuna study_name: {args.study_name} (load_if_exists=True)")
    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage_url,
        study_name=args.study_name,
        load_if_exists=True,
    )

    objective = make_objective(args)

    for t in range(args.n_trials):
        log(f"[TRIAL] {len(study.trials) + 1}/{len(study.trials) + (args.n_trials - t)} (incremental)")
        study.optimize(objective, n_trials=1, gc_after_trial=True)
        try:
            df_trials = study.trials_dataframe(
                attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete")
            )
            df_trials.to_csv(args.results_dir / "study_trials.csv", index=False)
        except Exception:
            pass

    best = study.best_trial
    best_params = dict(best.params)
    best_value = float(best.value)
    best_out = {
        "best_value_mae": best_value,
        "best_params": best_params,
        "n_trials_total": len(study.trials),
        "fixed": {
            "wind_features": args.wind_features,
            "aux_features": args.aux_features,
            "balance_strategy": args.balance_strategy,
            "gs_only": bool(args.gs_only),
            "n_jobs_outer": args.n_jobs_outer,
            "n_jobs_model": args.n_jobs_model,
            "seed": args.seed,
            "storage": storage_url,
            "study_name": args.study_name,
        },
    }
    with open(args.results_dir / "best_params.json", "w") as f:
        json.dump(best_out, f, indent=2)

    log(f"[BEST] MAE={best_value:.4f} with params: {best_params}")
    log(f"[INFO] Wrote best_params.json, hpo_manifest.csv, and study_trials.csv to {args.results_dir.resolve()}")
    log("[INFO] Done.")


if __name__ == "__main__":
    main()
