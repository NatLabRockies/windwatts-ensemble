#!/usr/bin/env python3
"""
Compare baseline vs experiment LOOCV predictions.

Loads baseline and one or more experiment prediction CSVs, merges on
(station_id, height_m, qnum), and reports:

  1. Overall metrics (pooled and mean-of-station)
  2. Per-quantile RMSE/MAE with winner tags
  3. Per-station RMSE/MAE with winner tags
  4. Optional hybrid analysis (experiment for low quantiles, baseline for upper tail)
  5. Stations driving divergence between aggregation methods

Usage examples::

    # Basic comparison
    wem-experiment compare \
        --baseline data/reference/loocv/ml_results.csv \
        --experiments data/output/enriched_results.csv

    # With custom label and hybrid analysis
    wem-experiment compare \
        --baseline data/reference/loocv/ml_results.csv \
        --experiments data/output/enriched_results.csv \
        --labels Enriched \
        --hybrid-cutoff 90

    # Compare concatenated batch files
    wem-experiment compare \
        --baseline data/reference/loocv/ml_results.csv \
        --experiments data/output/wide_batch*.csv \
        --labels Wide
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wem.utils.logging import log


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------


def load_and_merge(
    baseline_path: Path,
    experiment_paths: list[Path],
) -> pd.DataFrame:
    """Load baseline and experiment predictions, merge on (station_id, height_m, qnum).

    Returns a DataFrame with columns: station_id, height_m, qnum,
    obs, pred_base, pred_exp.
    """
    log(f"[INFO] Loading baseline: {baseline_path}")
    baseline = pd.read_csv(baseline_path, low_memory=False)

    log(f"[INFO] Loading {len(experiment_paths)} experiment prediction file(s)")
    exp_parts = [pd.read_csv(p, low_memory=False) for p in experiment_paths]
    exp = pd.concat(exp_parts, ignore_index=True)

    # Filter to rows with valid predictions (long-format outputs may include
    # NaN pred_observation for non-evaluated stations)
    baseline = baseline.dropna(subset=["pred_observation"])
    exp = exp.dropna(subset=["pred_observation"])

    stations = exp["station_id"].unique()
    log(f"[INFO] Experiment stations: {len(stations)}")

    baseline = baseline[baseline["station_id"].isin(stations)].copy()
    log(f"[INFO] Baseline rows after filter: {len(baseline)}")

    merged = baseline[["station_id", "height_m", "qnum", "observation", "pred_observation"]].rename(
        columns={"pred_observation": "pred_base", "observation": "obs"},
    ).merge(
        exp[["station_id", "height_m", "qnum", "pred_observation"]].rename(
            columns={"pred_observation": "pred_exp"},
        ),
        on=["station_id", "height_m", "qnum"],
        how="inner",
    )
    log(f"[INFO] Merged rows: {len(merged)}")
    return merged


def per_station_metrics(merged: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Compute RMSE and MAE per station for each prediction column.

    Returns a DataFrame indexed by station_id with columns
    ``{col}_rmse`` and ``{col}_mae`` for each col in *pred_cols*.
    """
    rows = []
    for sid, grp in merged.groupby("station_id"):
        obs = grp["obs"].values
        row = {"station_id": sid, "n_quantiles": len(grp)}
        for col in pred_cols:
            preds = grp[col].values
            diff = preds - obs
            row[f"{col}_rmse"] = float(np.sqrt(np.mean(diff**2)))
            row[f"{col}_mae"] = float(np.mean(np.abs(diff)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("station_id").sort_index()


def per_quantile_metrics(merged: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Compute RMSE and MAE per quantile for each prediction column.

    Returns a DataFrame indexed by qnum with columns
    ``{col}_rmse`` and ``{col}_mae`` for each col in *pred_cols*.
    """
    rows = []
    for q, grp in merged.groupby("qnum"):
        obs = grp["obs"].values
        row = {"qnum": int(q)}
        for col in pred_cols:
            preds = grp[col].values
            diff = preds - obs
            row[f"{col}_rmse"] = float(np.sqrt(np.mean(diff**2)))
            row[f"{col}_mae"] = float(np.mean(np.abs(diff)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("qnum").sort_index()


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_overall(merged: pd.DataFrame, pred_cols: list[str], labels: dict[str, str]):
    """Print overall metrics using both aggregation methods."""
    obs = merged["obs"].values
    station_df = per_station_metrics(merged, pred_cols)

    print()
    log("=" * 70)
    log(f"OVERALL METRICS ({merged['station_id'].nunique()} stations, {len(merged)} rows)")
    log("=" * 70)

    # Pooled
    print()
    log("Pooled (each quantile-row weighted equally):")
    base_rmse = None
    base_mae = None
    for col in pred_cols:
        preds = merged[col].values
        rmse = float(np.sqrt(np.mean((preds - obs) ** 2)))
        mae = float(np.mean(np.abs(preds - obs)))
        if base_rmse is None:
            base_rmse, base_mae = rmse, mae
            log(f"  {labels[col]:12s}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        else:
            rpct = (rmse - base_rmse) / base_rmse * 100
            mpct = (mae - base_mae) / base_mae * 100
            log(f"  {labels[col]:12s}  RMSE={rmse:.4f} ({rpct:+.1f}%)  MAE={mae:.4f} ({mpct:+.1f}%)")

    # Mean-of-station
    print()
    log("Mean-of-station (each station weighted equally):")
    base_rmse_s = None
    base_mae_s = None
    for col in pred_cols:
        rmse = station_df[f"{col}_rmse"].mean()
        mae = station_df[f"{col}_mae"].mean()
        if base_rmse_s is None:
            base_rmse_s, base_mae_s = rmse, mae
            log(f"  {labels[col]:12s}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        else:
            rpct = (rmse - base_rmse_s) / base_rmse_s * 100
            mpct = (mae - base_mae_s) / base_mae_s * 100
            log(f"  {labels[col]:12s}  RMSE={rmse:.4f} ({rpct:+.1f}%)  MAE={mae:.4f} ({mpct:+.1f}%)")


def print_per_quantile(merged: pd.DataFrame, pred_cols: list[str], labels: dict[str, str]):
    """Print per-quantile RMSE comparison."""
    qdf = per_quantile_metrics(merged, pred_cols)
    base_col = pred_cols[0]

    print()
    log("=" * 70)
    log("PER-QUANTILE RMSE")
    log("=" * 70)
    header = f"{'qnum':>5s}"
    for col in pred_cols:
        header += f"  {labels[col]:>10s}"
    header += f"  {'diff':>10s}  winner"
    log(header)
    log("-" * len(header))

    wins = {col: 0 for col in pred_cols}
    for q in range(101):
        row_str = f"q{q:03d} "
        rmses = {}
        for col in pred_cols:
            r = qdf.loc[q, f"{col}_rmse"]
            rmses[col] = r
            row_str += f"  {r:10.4f}"
        # Diff = last col vs first col
        diff = rmses[pred_cols[-1]] - rmses[base_col]
        best_col = min(rmses, key=rmses.get)
        wins[best_col] += 1
        row_str += f"  {diff:+10.4f}  {labels[best_col]}"
        log(row_str)

    log("-" * len(header))
    for col in pred_cols:
        log(f"  {labels[col]} wins at {wins[col]}/101 quantiles")


def print_per_station(
    merged: pd.DataFrame,
    pred_cols: list[str],
    labels: dict[str, str],
    top_n: int = 0,
):
    """Print per-station RMSE comparison.

    If *top_n* > 0, only show top/bottom N stations instead of all.
    """
    sdf = per_station_metrics(merged, pred_cols)
    base_col = pred_cols[0]

    print()
    log("=" * 70)
    log(f"PER-STATION RMSE ({len(sdf)} stations)")
    log("=" * 70)

    header = f"{'station':>25s}"
    for col in pred_cols:
        header += f"  {labels[col]:>8s}"
    header += f"  {'diff':>10s}  best"
    log(header)
    log("-" * len(header))

    sdf["_diff"] = sdf[f"{pred_cols[-1]}_rmse"] - sdf[f"{base_col}_rmse"]
    sdf_sorted = sdf.sort_values("_diff")

    if top_n > 0 and len(sdf_sorted) > 2 * top_n:
        display = pd.concat([sdf_sorted.head(top_n), sdf_sorted.tail(top_n)])
        show_sep = True
    else:
        display = sdf_sorted
        show_sep = False

    wins = {col: 0 for col in pred_cols}
    shown_sep = False
    for sid, row in sdf_sorted.iterrows():
        rmses = {col: row[f"{col}_rmse"] for col in pred_cols}
        best_col = min(rmses, key=rmses.get)
        wins[best_col] += 1

        if sid not in display.index:
            continue
        if show_sep and not shown_sep and row["_diff"] > 0:
            log(f"{'... ':>25s}")
            shown_sep = True

        line = f"{sid:>25s}"
        for col in pred_cols:
            line += f"  {rmses[col]:8.4f}"
        line += f"  {row['_diff']:+10.4f}  {labels[best_col]}"
        log(line)

    log("-" * len(header))
    for col in pred_cols:
        log(f"  {labels[col]} wins at {wins[col]}/{len(sdf)} stations")


def print_divergence(merged: pd.DataFrame, pred_cols: list[str], labels: dict[str, str], n: int = 5):
    """Show stations driving divergence between pooled and mean-of-station metrics."""
    if len(pred_cols) < 2:
        return

    sdf = per_station_metrics(merged, pred_cols)
    base_col, comp_col = pred_cols[0], pred_cols[-1]
    sdf["_diff"] = sdf[f"{comp_col}_rmse"] - sdf[f"{base_col}_rmse"]

    print()
    log("=" * 70)
    log("STATIONS DRIVING DIVERGENCE")
    log("=" * 70)
    log(f"Biggest wins for {labels[comp_col]} (pull pooled RMSE down):")
    for sid, row in sdf.nsmallest(n, "_diff").iterrows():
        log(f"  {sid:25s}  base={row[f'{base_col}_rmse']:.4f}  "
            f"comp={row[f'{comp_col}_rmse']:.4f}  diff={row['_diff']:+.4f}")
    log(f"Biggest losses for {labels[comp_col]} (pull mean-station RMSE up):")
    for sid, row in sdf.nlargest(n, "_diff").iterrows():
        log(f"  {sid:25s}  base={row[f'{base_col}_rmse']:.4f}  "
            f"comp={row[f'{comp_col}_rmse']:.4f}  diff={row['_diff']:+.4f}")


# ---------------------------------------------------------------------------
# Runner entry point
# ---------------------------------------------------------------------------


def run_compare(args):
    """Run comparison analysis from pre-parsed args namespace.

    Expected attributes: baseline, experiments, labels, hybrid_cutoff,
    top_n, save_csv, quantile_detail.
    """
    merged = load_and_merge(args.baseline, args.experiments)

    pred_cols = ["pred_base", "pred_exp"]

    # Determine labels
    if args.labels:
        exp_label = args.labels[0]
    else:
        # Default: derive from first experiment filename stem
        exp_label = args.experiments[0].stem
    labels = {"pred_base": "Baseline", "pred_exp": exp_label}

    # Optional hybrid
    if args.hybrid_cutoff is not None:
        q = args.hybrid_cutoff
        merged["pred_hybrid"] = np.where(
            merged["qnum"] < q, merged["pred_exp"], merged["pred_base"],
        )
        pred_cols.append("pred_hybrid")
        labels["pred_hybrid"] = f"Hybrid(<{q})"
        log(f"[INFO] Hybrid model: experiment for q000-q{q-1:03d}, baseline for q{q:03d}-q100")

    # Reports
    print_overall(merged, pred_cols, labels)
    if args.quantile_detail:
        print_per_quantile(merged, pred_cols, labels)
    else:
        # Summary: show every 10th quantile
        qdf = per_quantile_metrics(merged, pred_cols)
        print()
        log("PER-QUANTILE RMSE (every 10th; use --quantile-detail for all 101):")
        base_col = pred_cols[0]
        last_col = pred_cols[-1]
        exp_wins = sum(
            1 for q in range(101)
            if qdf.loc[q, f"{last_col}_rmse"] < qdf.loc[q, f"{base_col}_rmse"]
        )
        for q in range(0, 101, 10):
            b = qdf.loc[q, f"{base_col}_rmse"]
            w = qdf.loc[q, f"{last_col}_rmse"]
            winner = labels[last_col] if w < b else labels[base_col]
            log(f"  q{q:03d}: {labels[base_col]}={b:.4f}  {labels[last_col]}={w:.4f}  "
                f"diff={w - b:+.4f}  [{winner}]")
        log(f"  {labels[last_col]} wins at {exp_wins}/101 quantiles")

    print_per_station(merged, pred_cols, labels, top_n=args.top_n)
    print_divergence(merged, pred_cols, labels)

    # Save CSV
    if args.save_csv:
        sdf = per_station_metrics(merged, pred_cols)
        sdf.to_csv(args.save_csv)
        log(f"[INFO] Saved per-station comparison → {args.save_csv}")

    log("[INFO] Done.")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare baseline vs experiment LOOCV predictions.",
    )
    ap.add_argument(
        "--baseline", type=Path, required=True,
        help="Baseline long-format predictions CSV (e.g. ml_results.csv).",
    )
    ap.add_argument(
        "--experiments", type=Path, nargs="+", required=True,
        help="One or more experiment prediction CSVs to concatenate.",
    )
    ap.add_argument(
        "--labels", type=str, nargs="*", default=None,
        help="Display labels for experiment files (default: derived from filenames).",
    )
    ap.add_argument(
        "--hybrid-cutoff", type=int, default=None, metavar="Q",
        help="If set, add hybrid model using experiment for q < Q and baseline for q >= Q.",
    )
    ap.add_argument(
        "--top-n", type=int, default=0,
        help="If > 0, only show top/bottom N stations per table (default: show all).",
    )
    ap.add_argument(
        "--save-csv", type=Path, default=None,
        help="Save per-station comparison table to this CSV.",
    )
    ap.add_argument(
        "--quantile-detail", action="store_true",
        help="Print full per-quantile table (101 rows). Default: summary only.",
    )
    args = ap.parse_args(argv)
    run_compare(args)


if __name__ == "__main__":
    main()
