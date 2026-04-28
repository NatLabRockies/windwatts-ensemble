"""Feature combination sweep runner.

Enumerates subsets of either wind or auxiliary features and invokes the
LOOCV training script for each combination. Consolidates the former
``sweep_wind_features.py`` and ``sweep_aux_features.py`` dev scripts.

Example (wind)::

    wem-exp-feature-sweep \\
        --feature-type wind \\
        --infile combined_quantiles_long_with_topo_loocv_10km.csv \\
        --outdir results_wind_sweep

Example (aux)::

    wem-exp-feature-sweep \\
        --feature-type aux \\
        --infile combined_quantiles_long_with_topo_loocv_10km.csv \\
        --outdir results_aux_sweep \\
        --capture-logs
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd

from wem.constants import AUX_GROUPS, WIND_FEATURES
from wem.experiment._helpers import (
    build_train_cmd,
    combo_label,
    enumerate_subsets,
    resolve_train_command,
)
from wem.utils.logging import log


def main():
    pa = argparse.ArgumentParser(
        description="Run training script for all feature subset combinations."
    )
    pa.add_argument(
        "--feature-type", type=str, required=True, choices=["wind", "aux"],
        help="Which feature group to sweep.",
    )
    pa.add_argument(
        "--features", type=str, default=None,
        help="Comma-separated feature list. Defaults to WIND_FEATURES or AUX_GROUPS.",
    )
    pa.add_argument(
        "--fixed-wind-features", type=str, default="hrrr,wtk,wtk_led_conus",
        help="Fixed wind features when --feature-type=aux.",
    )
    pa.add_argument("--include-empty", action="store_true")
    pa.add_argument(
        "--capture-logs", action="store_true",
        help="Write per-run stdout/stderr to files.",
    )

    pa.add_argument(
        "--train-cmd", type=str, default=None,
        help="Training command override. Default: python -m wem.train.loocv_xgb",
    )
    pa.add_argument(
        "--infile", type=Path,
        default=Path("combined_quantiles_long_with_topo_loocv_10km.csv"),
    )
    pa.add_argument("--outdir", type=Path, default=None)
    pa.add_argument("--balance_strategy", type=str, default="upsample",
                    choices=["downsample", "upsample"])
    pa.add_argument("--gs_only", action="store_true", default=True)
    pa.add_argument("--n_jobs_outer", type=int, default=None)
    pa.add_argument("--n_jobs_model", type=int, default=None)
    pa.add_argument("--seed", type=int, default=None)
    pa.add_argument("--extra_args", type=str, default="")
    args = pa.parse_args()

    # Determine feature list
    if args.features:
        feature_list = [f.strip() for f in args.features.split(",") if f.strip()]
    elif args.feature_type == "wind":
        feature_list = list(WIND_FEATURES)
    else:
        feature_list = list(AUX_GROUPS)

    combos = enumerate_subsets(feature_list, include_empty=args.include_empty)

    # Results directory
    if args.outdir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"feature_sweeps_{args.feature_type}_{stamp}")
    else:
        outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"[INFO] Writing results to: {outdir.resolve()}")

    train_cmd = resolve_train_command(args.train_cmd)
    manifest_rows = []

    # Determine file prefix based on feature type
    prefix = "ml_results_xgb_" if args.feature_type == "wind" else "ml_results_xgb_aux_"

    for feats in combos:
        label = combo_label(feats)
        outfile = outdir / f"{prefix}{label}.csv"

        if args.feature_type == "wind":
            wind_str = ",".join(feats)
            aux_str = None
        else:
            wind_str = args.fixed_wind_features
            aux_str = ",".join(feats) if feats else ""

        cmd = build_train_cmd(
            train_cmd,
            args.infile,
            outfile,
            wind_features=wind_str,
            aux_features=aux_str,
            balance_strategy=args.balance_strategy,
            gs_only=args.gs_only,
            n_jobs_outer=args.n_jobs_outer,
            n_jobs_model=args.n_jobs_model,
            seed=args.seed if args.seed is not None else 42,
            extra_args=args.extra_args,
        )

        log(f"[RUN] {label} -> {outfile.name}")
        t0 = time.time()

        if args.capture_logs:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            dt = round(time.time() - t0, 2)
            (outdir / f"log_{label}.stdout.txt").write_text(proc.stdout or "")
            (outdir / f"log_{label}.stderr.txt").write_text(proc.stderr or "")
            status = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"
        else:
            try:
                subprocess.run(cmd, check=True)
                status = "ok"
            except subprocess.CalledProcessError as e:
                log(f"[FAIL] {label}: returncode={e.returncode}")
                status = f"fail:{e.returncode}"
            dt = round(time.time() - t0, 2)

        manifest_rows.append({
            "label": label,
            "features": ",".join(feats) if feats else "(none)",
            "outfile": str(outfile),
            "status": status,
            "seconds": dt,
        })

    man_path = outdir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(man_path, index=False)
    log(f"[INFO] Wrote manifest -> {man_path}")
    log("[INFO] All runs complete.")


if __name__ == "__main__":
    main()
