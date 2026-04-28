"""Row-level error metrics for wind datasets and ML ensemble.

Row-level RMSE, MAE, and bias for public wind datasets and the ML ensemble
on GS quantile rows.

Input: Long-format quantile table (CSV or Parquet)
Output: row_metrics_gs.csv + LaTeX table rows
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from wem.constants import DATASET_LATEX
from wem.utils.logging import log
from wem.utils.sites import normalize_obs_type


def compute_row_metrics(
    df: pd.DataFrame,
    obs_col: str,
    dataset_cols: List[str],
    subset: str = "GS",
) -> pd.DataFrame:
    """Compute RMSE_row, MAE_row, and bias_row for each dataset.

    Parameters
    ----------
    df : DataFrame
        Long-format quantile table with observation and dataset columns.
    obs_col : str
        Column name for observed quantiles.
    dataset_cols : list of str
        Candidate dataset column names (only those present in df are used).
    subset : {'GS', 'ASOS', 'ALL'}
        Which observation_type subset to use.

    Returns
    -------
    metrics : DataFrame
        Columns: dataset, subset, n_rows, rmse_row, mae_row, bias_row
    """
    needed = ["observation_type", obs_col]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Input missing required column: {c!r}")

    df = df.copy()
    df["observation_type"] = (
        df["observation_type"].astype(str).map(normalize_obs_type)
    )

    if subset.upper() in {"GS", "ASOS"}:
        df = df[df["observation_type"] == subset.upper()].copy()
        log(f"[INFO] Subset={subset.upper()} -> {len(df):,} rows")
    else:
        log("[INFO] Using ALL rows (no observation_type filter).")

    df[obs_col] = pd.to_numeric(df[obs_col], errors="coerce")

    dataset_cols = [c for c in dataset_cols if c in df.columns]
    if not dataset_cols:
        raise ValueError(
            "None of the requested dataset columns are present in the input."
        )

    records: List[Dict] = []

    for d in dataset_cols:
        x = pd.to_numeric(df[d], errors="coerce")
        y = df[obs_col]

        good = np.isfinite(x) & np.isfinite(y)
        if not np.any(good):
            log(f"[WARN] No valid rows for dataset={d}; skipping.")
            continue

        e = (x[good] - y[good]).to_numpy(dtype="float64")
        n = e.size
        rmse = float(np.sqrt(np.mean(e**2)))
        mae = float(np.mean(np.abs(e)))
        bias = float(np.mean(e))

        records.append(
            {
                "dataset": d,
                "subset": subset.upper(),
                "n_rows": int(n),
                "rmse_row": rmse,
                "mae_row": mae,
                "bias_row": bias,
            }
        )
        log(
            f"[INFO] {d}: n_rows={n:,}, RMSE_row={rmse:.4f}, "
            f"MAE_row={mae:.4f}, BIAS_row={bias:.4f}"
        )

    return pd.DataFrame(records)


def print_latex_table(
    metrics: pd.DataFrame, dataset_order: List[str]
) -> None:
    """Print LaTeX rows for Table~\\ref{tab:ml-vs-baselines}."""

    def fmt(x: float) -> str:
        if x is None or not np.isfinite(x):
            return r"\dots"
        return f"{x:.3f}"

    subset_vals = metrics["subset"].unique()
    if len(subset_vals) != 1:
        log(f"[WARN] Expected a single subset, found: {subset_vals}")
    subset = subset_vals[0] if len(subset_vals) else "GS"

    print("% LaTeX rows for Table~\\ref{tab:ml-vs-baselines}")
    print("% Subset:", subset)
    for d in dataset_order:
        row = metrics[metrics["dataset"] == d]
        if row.empty:
            continue
        r0 = row.iloc[0]
        label = DATASET_LATEX.get(d, d)
        rmse = fmt(r0["rmse_row"])
        mae = fmt(r0["mae_row"])
        bias = fmt(r0["bias_row"])
        print(f"{label:20s} & {rmse} & {mae} & {bias} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Row-level RMSE_row and MAE_row for GS quantile rows."
    )
    ap.add_argument(
        "--infile",
        type=Path,
        required=True,
        help="Long-format quantile table (CSV or Parquet).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_out_row_metrics"),
        help="Directory for row_metrics_gs.csv output.",
    )
    ap.add_argument(
        "--obs_col",
        type=str,
        default="observation",
        help="Column name for observed quantiles.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="era5,wtk,wtk_led_climate,wtk_led_conus,hrrr,pred_observation",
        help="Comma-separated list of dataset column names to evaluate.",
    )
    ap.add_argument(
        "--subset",
        type=str,
        default="GS",
        choices=["GS", "ASOS", "ALL"],
        help="Observation subset to use for metrics (default: GS).",
    )
    args = ap.parse_args()

    if not args.infile.exists():
        raise FileNotFoundError(args.infile)
    args.outdir.mkdir(parents=True, exist_ok=True)

    dataset_cols = [
        c.strip() for c in args.datasets.split(",") if c.strip()
    ]
    log(f"[INFO] Requested dataset columns: {dataset_cols}")

    log(f"[INFO] Loading long table: {args.infile}")
    if args.infile.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.infile)
    else:
        df = pd.read_csv(
            args.infile, low_memory=False, dtype={"station_id": str}
        )

    log("[INFO] Computing row-level metrics ...")
    metrics = compute_row_metrics(
        df,
        obs_col=args.obs_col,
        dataset_cols=dataset_cols,
        subset=args.subset,
    )

    out_csv = args.outdir / f"row_metrics_{args.subset.lower()}.csv"
    metrics.to_csv(out_csv, index=False)
    log(f"[INFO] Wrote row-level metrics -> {out_csv}")

    default_order = [
        "era5",
        "wtk",
        "wtk_led_climate",
        "wtk_led_conus",
        "hrrr",
        "gwa",
        "pred_observation",
    ]
    present = [
        d for d in default_order if d in metrics["dataset"].unique()
    ]
    print_latex_table(metrics, dataset_order=present)

    log("[INFO] Done.")


if __name__ == "__main__":
    main()
