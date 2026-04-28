"""Grid sweep of a single numeric XGBoost hyperparameter.

Replaces the separate ``optimize_n_estimators.py`` and ``optimize_max_depth.py``
dev scripts with a single parameterized module. Sweeps any numeric XGBoost
hyperparameter over a start/stop/step range, running the LOOCV training script
for each value.

Example::

    wem-exp-param-sweep \\
        --param n_estimators --start 200 --stop 1000 --step 100 \\
        --infile combined_quantiles_long_with_topo_loocv_10km.csv \\
        --results-dir results_n_estimators \\
        --gs_only
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from wem.experiment._helpers import (
    build_train_cmd,
    compute_site_mae,
    resolve_train_command,
)
from wem.utils.logging import log

INTEGER_PARAMS = {"n_estimators", "max_depth"}


def main():
    pa = argparse.ArgumentParser(
        description="Serial sweep of a numeric XGBoost hyperparameter."
    )
    pa.add_argument(
        "--param", type=str, required=True,
        choices=["n_estimators", "max_depth", "learning_rate",
                 "min_child_weight", "subsample", "colsample_bytree"],
        help="The hyperparameter to sweep.",
    )
    pa.add_argument("--start", type=float, required=True, help="Start value (inclusive).")
    pa.add_argument("--stop", type=float, required=True, help="Stop value (inclusive).")
    pa.add_argument("--step", type=float, required=True, help="Step size.")

    pa.add_argument(
        "--train-cmd", type=str, default=None,
        help="Training command override. Default: python -m wem.train.loocv_xgb",
    )
    pa.add_argument(
        "--infile", type=Path,
        default=Path("combined_quantiles_long_with_topo_loocv_10km.csv"),
    )
    pa.add_argument("--results-dir", type=Path, default=Path("results_param_sweep"))
    pa.add_argument("--seed", type=int, default=42)

    pa.add_argument("--wind_features", type=str, default="hrrr,wtk,wtk_led_conus")
    pa.add_argument("--aux_features", type=str, default="latlon,height,elevation")
    pa.add_argument(
        "--balance_strategy", type=str, default="downsample",
        choices=["downsample", "upsample"],
    )
    pa.add_argument("--gs_only", action="store_true")
    pa.add_argument("--n_jobs_outer", type=int, default=12)
    pa.add_argument("--n_jobs_model", type=int, default=1)

    # Non-swept hyperparams (defaults match production config)
    pa.add_argument("--learning_rate", type=float, default=0.02216030268952961)
    pa.add_argument("--max_depth", type=int, default=20)
    pa.add_argument("--n_estimators", type=int, default=500)
    pa.add_argument("--min_child_weight", type=float, default=4.2832509812996635)
    pa.add_argument("--subsample", type=float, default=0.6098353951742953)
    pa.add_argument("--colsample_bytree", type=float, default=0.9761640794652597)

    pa.add_argument("--extra_args", type=str, default="")
    args = pa.parse_args()
    args.results_dir = getattr(args, "results_dir", None) or Path("results_param_sweep")

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = resolve_train_command(args.train_cmd)

    swept = args.param
    is_int = swept in INTEGER_PARAMS

    # Build sweep values
    raw_vals = np.arange(args.start, args.stop + args.step * 0.5, args.step)
    if is_int:
        sweep_vals = [int(v) for v in raw_vals if v > 0]
    else:
        sweep_vals = [float(v) for v in raw_vals if v > 0]

    manifest_rows: List[dict] = []
    site_rows: List[dict] = []
    summary_rows: List[dict] = []

    best_val = None
    best_mean = float("inf")
    best_median = float("inf")

    log(f"[INFO] Starting serial sweep of {swept}; early stopping disabled.")
    log(f"[INFO] Range: {args.start}..{args.stop} step {args.step}")
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for sv in sweep_vals:
        sv_str = str(sv) if is_int else f"{sv:.6f}"
        tag = f"{sv:04d}" if is_int else f"{sv:.4f}"
        outfile = results_dir / f"ml_results_xgb_{swept}_{tag}.csv"
        logfile = results_dir / f"ml_results_xgb_{swept}_{tag}.log"

        # Build params dict: start from CLI defaults, override swept param
        params = {
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
        params[swept] = sv

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

        need_run = not outfile.exists()
        ret = 0
        dur = 0.0
        if need_run:
            with open(logfile, "w") as lf:
                lf.write(f"# {swept}={sv}\n")
                lf.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
                lf.flush()
                start = time.time()
                proc = subprocess.run(cmd, stdout=lf, stderr=lf, check=False)
                dur = time.time() - start
                ret = proc.returncode
                lf.write(f"\n[DONE] returncode={ret} duration_sec={dur:.2f}\n")
        else:
            log(f"[SKIP] Output exists for {swept}={sv}: {outfile.name}")

        manifest_rows.append({
            "timestamp": stamp,
            swept: sv,
            "outfile": str(outfile),
            "logfile": str(logfile),
            "returncode": ret,
            "duration_sec": dur,
        })

        site_df, mean_mae, median_mae, n_sites = compute_site_mae(outfile)
        if site_df.empty or not np.isfinite(mean_mae):
            log(f"[WARN] Could not compute MAE for {swept}={sv}.")
        else:
            for _, r in site_df.iterrows():
                site_rows.append({
                    swept: sv,
                    "station_id": str(r["station_id"]),
                    "mae": float(r["mae"]),
                })

        summary_rows.append({
            swept: sv,
            "mae_mean": mean_mae,
            "mae_median": median_mae,
            "n_stations": n_sites,
            "outfile": str(outfile),
        })

        mm = "nan" if not np.isfinite(mean_mae) else f"{mean_mae:.4f}"
        md = "nan" if not np.isfinite(median_mae) else f"{median_mae:.4f}"
        log(f"[ITER] {swept}={sv}  mean MAE={mm}  median MAE={md}")

        if np.isfinite(mean_mae):
            if (mean_mae < best_mean) or (
                np.isfinite(best_mean)
                and mean_mae == best_mean
                and np.isfinite(median_mae)
                and median_mae < best_median
            ):
                best_mean = mean_mae
                best_median = median_mae if np.isfinite(median_mae) else best_median
                best_val = sv
                log(f"[BEST] Updated best: {swept}={best_val}  mean MAE={best_mean:.4f}")

        # Persist incremental CSVs
        pd.DataFrame(manifest_rows).to_csv(results_dir / "manifest.csv", index=False)
        pd.DataFrame(site_rows).to_csv(
            results_dir / f"site_mae_by_{swept}.csv", index=False
        )
        pd.DataFrame(summary_rows).to_csv(
            results_dir / f"summary_by_{swept}.csv", index=False
        )

    if best_val is not None:
        # Remove swept param from fixed hyperparams
        fixed_hp = {
            k: v for k, v in params.items() if k != swept
        }
        best_payload = {
            f"best_{swept}": best_val,
            "best_mean_mae": best_mean,
            "best_median_mae": best_median,
            "range": {"start": args.start, "stop": args.stop, "step": args.step},
            "fixed_hyperparams": fixed_hp,
            "config": {
                "wind_features": args.wind_features,
                "aux_features": args.aux_features,
                "balance_strategy": args.balance_strategy,
                "gs_only": bool(args.gs_only),
                "seed": args.seed,
            },
        }
        with open(results_dir / f"best_{swept}.json", "w") as f:
            json.dump(best_payload, f, indent=2)

        log(f"[FINAL] Best {swept}={best_val}  mean MAE={best_mean:.4f}")
        log(f"[INFO] Wrote results to {results_dir.resolve()}")
    else:
        log("[ERROR] No valid runs produced finite MAE.")


if __name__ == "__main__":
    main()
