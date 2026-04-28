"""Analyze feature-sweep results and generate diagnostic visualizations.

Consolidates the former ``analyze_feature_sweeps.py`` (wind) and
``analyze_aux_sweeps.py`` (aux) dev scripts into a single parameterized
module. Produces metric summaries, marginal delta analyses, and plots.

Example::

    wem-exp-analyze-sweep \\
        --results-dir results_wind_sweep \\
        --feature-type wind \\
        --metric rmse \\
        --top-k 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from wem.constants import AUX_GROUPS, WIND_FEATURES
from wem.experiment._helpers import (
    ci95,
    compute_sweep_metrics,
    features_from_label,
    make_pairs_for_feature,
    parse_label_from_path,
)
from wem.utils.logging import log

# --------------- plot configuration ---------------

WIND_PLOT_CONFIG: dict = {
    "feature_labels": {
        "era5": "ERA5",
        "wtk": "WTK",
        "hrrr": "HRRR",
        "wtk_led_conus": "WTK-LED CONUS",
        "wtk_led_climate": "WTK-LED Climate",
    },
    "box_order": ["WTK", "HRRR", "WTK-LED CONUS", "WTK-LED Climate", "ERA5"],
    "box_palette": ["#F28E2B", "#E15759", "#B07AA1", "#76B7B2", "#4E79A7"],
    "heatmap_mode": "colored",
    "heatmap_palette": {
        "ERA5": "#4E79A7",
        "WTK": "#F28E2B",
        "HRRR": "#E15759",
        "WTK-LED CONUS": "#B07AA1",
        "WTK-LED Climate": "#76B7B2",
    },
    "metric_xlabel": "# wind features used",
    "file_prefix": "ml_results_xgb_",
}

AUX_PLOT_CONFIG: dict = {
    "feature_labels": {f: f for f in AUX_GROUPS},
    "box_order": list(AUX_GROUPS),
    "box_palette": ["#F7D7C4"] * len(AUX_GROUPS),
    "box_edge_color": "#8A3A2B",
    "heatmap_mode": "grayscale",
    "metric_xlabel": "# aux features used",
    "file_prefix": "ml_results_xgb_aux_",
}


# --------------- plotting ---------------

def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def savefig(fig, outpath: Path, dpi: int = 300):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    _import_mpl().close(fig)


def plot_overall_bar(top_df: pd.DataFrame, key_metric: str, out: Path, dpi: int = 300):
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top_df))))
    ax.barh(range(len(top_df)), top_df[key_metric], color="#4E79A7")
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel(key_metric.upper())
    ax.set_title(f"Top combinations by {key_metric.upper()}")
    savefig(fig, out, dpi)


def plot_metric_by_count(
    summary: pd.DataFrame, key_metric: str, out: Path,
    xlabel: str = "# features used", dpi: int = 300,
):
    plt = _import_mpl()
    counts = sorted(summary["n_feats"].unique())
    data = [summary.loc[summary["n_feats"] == k, key_metric].dropna().values for k in counts]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=[str(k) for k in counts], showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set(facecolor="#D3E1F3", edgecolor="#355C8B")
    for w in bp["whiskers"]:
        w.set(color="#355C8B")
    for c in bp["caps"]:
        c.set(color="#355C8B")
    for m in bp["medians"]:
        m.set(color="#1f1f1f")

    for i, y in enumerate(data, start=1):
        if y.size == 0:
            continue
        xs = np.random.default_rng(0).uniform(i - 0.25, i + 0.25, size=y.size)
        ax.plot(xs, y, "o", ms=3, alpha=0.6, color="#4E79A7")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(key_metric.upper())
    ax.set_title(f"{key_metric.upper()} by number of features")
    savefig(fig, out, dpi)


def plot_marginal_box(
    deltas: pd.DataFrame, key_metric: str, out: Path,
    config: dict, dpi: int = 300,
):
    plt = _import_mpl()
    labels_map = config.get("feature_labels", {})
    box_order = config.get("box_order", [])
    box_palette = config.get("box_palette", [])
    edge_color = config.get("box_edge_color", "black")

    # Map raw feature names to pretty labels
    inv_labels = {v: k for k, v in labels_map.items()}
    present_pretty = [lbl for lbl in box_order if inv_labels.get(lbl, lbl) in deltas["feature"].unique()]

    data = []
    for lbl in present_pretty:
        raw = inv_labels.get(lbl, lbl)
        data.append(deltas.loc[deltas["feature"] == raw, "delta"].dropna().values)

    palette_map = dict(zip(box_order, box_palette))
    colors = [palette_map.get(lbl, "#CCCCCC") for lbl in present_pretty]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=present_pretty, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, edgecolor=edge_color, linewidth=1.3)
    for w in bp["whiskers"]:
        w.set(color=edge_color)
    for c in bp["caps"]:
        c.set(color=edge_color)
    for m in bp["medians"]:
        m.set(color="black", linewidth=1.5)

    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(f"Delta {key_metric.upper()}  (with feature - without feature)")
    savefig(fig, out, dpi)


def plot_marginal_bar_means(
    deltas: pd.DataFrame, key_metric: str, out: Path,
    feature_list: list[str], dpi: int = 300,
):
    plt = _import_mpl()
    feats = [f for f in feature_list if (deltas["feature"] == f).any()]
    rows = []
    for f in feats:
        x = deltas.loc[deltas["feature"] == f, "delta"].dropna().to_numpy()
        if x.size == 0:
            continue
        mu = float(np.mean(x))
        lo, hi = ci95(x)
        rows.append({"feature": f, "mean": mu, "lo": lo, "hi": hi, "n": int(x.size)})

    if not rows:
        return

    dfc = pd.DataFrame(rows).set_index("feature").reindex(feats)

    lower = np.nan_to_num((dfc["mean"] - dfc["lo"]).values, nan=0.0)
    upper = np.nan_to_num((dfc["hi"] - dfc["mean"]).values, nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(dfc))
    ax.barh(y_pos, dfc["mean"].values, xerr=[lower, upper],
            color="#E15759", alpha=0.9, ecolor="#333", capsize=4)

    for i, n in enumerate(dfc["n"].values):
        ax.text(dfc["mean"].values[i], y_pos[i], f"  n={n}",
                va="center", ha="left", fontsize=8, color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(dfc.index.tolist())
    ax.axvline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(f"Mean Delta {key_metric.upper()} (with - without)")
    ax.set_title("Average marginal effect per feature (95% CI)")
    savefig(fig, out, dpi)


def plot_combo_presence_heatmap(
    top_df: pd.DataFrame, out: Path,
    feature_list: list[str], config: dict, key_metric: str = "mae",
    dpi: int = 300,
):
    plt = _import_mpl()
    if top_df.empty:
        return

    n_rows = len(top_df)
    n_cols = len(feature_list)

    presence = np.zeros((n_rows, n_cols), dtype=int)
    for i, feats in enumerate(top_df["feats"]):
        for j, f in enumerate(feature_list):
            if f in feats:
                presence[i, j] = 1

    labels_map = config.get("feature_labels", {})
    pretty_labels = [labels_map.get(f, f) for f in feature_list]
    mode = config.get("heatmap_mode", "grayscale")

    fig = plt.figure(figsize=(8, max(4, 0.4 * n_rows)))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2, 1], wspace=0.05)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    if mode == "colored":
        heatmap_pal = config.get("heatmap_palette", {})
        col_colors = [heatmap_pal.get(lbl, "#CCCCCC") for lbl in pretty_labels]

        ax_h.set_xlim(0, n_cols)
        ax_h.set_ylim(0, n_rows)
        for i in range(n_rows):
            for j in range(n_cols):
                face = col_colors[j] if presence[i, j] == 1 else "#FFFFFF"
                ax_h.add_patch(plt.Rectangle(
                    (j, n_rows - 1 - i), 1, 1,
                    facecolor=face, edgecolor="white",
                ))
        ax_h.set_yticks([])
        ax_h.set_xticks(np.arange(n_cols) + 0.5)
        ax_h.set_xticklabels(pretty_labels, rotation=30, ha="right")
    else:
        ax_h.imshow(presence, aspect="auto", cmap=plt.cm.Greys, vmin=0, vmax=1)
        ax_h.set_yticks(range(n_rows))
        ax_h.set_yticklabels(top_df["label"])
        ax_h.set_xticks(range(n_cols))
        ax_h.set_xticklabels(pretty_labels, rotation=30, ha="right")

    ax_h.set_title("Feature Presence")

    # Bar chart
    metric_col = key_metric if key_metric in top_df.columns else "rmse"
    y_pos = np.arange(n_rows)
    ax_b.barh(y_pos, top_df[metric_col].values, color="#AAAAAA", height=1, edgecolor="#FFFFFF")
    ax_b.set_yticks([])
    ax_b.invert_yaxis()
    ax_b.set_xlabel(metric_col.upper())
    ax_b.set_title("Model Error")

    savefig(fig, out, dpi)


# --------------- main analysis logic ---------------

def analyze_sweep_results(
    results_dir: Path,
    feature_list: list[str],
    config: dict,
    key_metric: str = "rmse",
    stationwise: bool = False,
    top_k: int = 12,
    dpi: int = 300,
) -> Path:
    """Run full sweep analysis and return the output directory."""
    outdir = results_dir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"[INFO] Writing analysis to: {outdir.resolve()}")

    file_prefix = config.get("file_prefix", "ml_results_xgb_")

    # Discover runs
    manifest_path = results_dir / "manifest.csv"
    runs = []
    if manifest_path.exists():
        log("[INFO] Using manifest.csv")
        man = pd.read_csv(manifest_path)
        for _, r in man.iterrows():
            p = Path(r.get("outfile", ""))
            if p.exists() and p.suffix.lower() == ".csv":
                label = parse_label_from_path(p, prefix=file_prefix)
                runs.append((label, p))
    else:
        log("[INFO] No manifest.csv - globbing outputs")
        for p in results_dir.glob(f"{file_prefix}*.csv"):
            label = parse_label_from_path(p, prefix=file_prefix)
            runs.append((label, p))

    if not runs:
        raise SystemExit(f"No {file_prefix}*.csv files found.")

    # Compute metrics
    metrics_rows = []
    for label, path in runs:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            log(f"[WARN] Failed to read {path.name}: {e}")
            continue
        if not {"observation", "pred_observation", "observation_type"}.issubset(df.columns):
            log(f"[WARN] Missing required columns in {path.name}; skipping.")
            continue

        m = compute_sweep_metrics(df, stationwise=stationwise)
        _, feats = features_from_label(label)
        metrics_rows.append({
            "label": label,
            "feats": feats,
            "n_feats": len(feats),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "rmse_stationwise": m["rmse_stationwise"],
            "mae_stationwise": m["mae_stationwise"],
            "rows": m["rows"],
            "stations": m["stations"],
            "file": str(path),
        })

    if not metrics_rows:
        raise SystemExit("No valid runs with metrics.")

    summary = pd.DataFrame(metrics_rows)
    summary = summary.sort_values(key_metric, ascending=True, na_position="last").reset_index(drop=True)
    summary.to_csv(outdir / "metrics_summary.csv", index=False)
    log("[INFO] Wrote metrics_summary.csv")

    # Marginal deltas
    by_set: Dict[Tuple[str, ...], dict[str, float]] = {}
    for _, r in summary.iterrows():
        by_set[tuple(sorted(r["feats"]))] = {
            "rmse": float(r["rmse"]) if np.isfinite(r["rmse"]) else np.nan,
            "mae": float(r["mae"]) if np.isfinite(r["mae"]) else np.nan,
        }

    delta_rows = []
    for f in feature_list:
        pairs = make_pairs_for_feature(by_set, f, key_metric=key_metric)
        for S, d in pairs:
            delta_rows.append({
                "feature": f,
                "base_set": "+".join(S) if S else "none",
                "delta": d,
            })

    if delta_rows:
        deltas_df = pd.DataFrame(delta_rows)
        deltas_df.to_csv(outdir / "marginal_deltas.csv", index=False)
        log("[INFO] Wrote marginal_deltas.csv")
    else:
        deltas_df = pd.DataFrame(columns=["feature", "base_set", "delta"])
        log("[WARN] No paired runs for marginal deltas.")

    # Plots
    top_df = summary.head(top_k).copy()
    plot_overall_bar(top_df[["label", key_metric]], key_metric,
                     outdir / "overall_rmse_by_combo.png", dpi)
    plot_metric_by_count(summary[["n_feats", key_metric]].copy(), key_metric,
                         outdir / "rmse_by_num_features.png",
                         xlabel=config.get("metric_xlabel", "# features used"),
                         dpi=dpi)

    if not deltas_df.empty:
        plot_marginal_box(deltas_df.copy(), key_metric,
                          outdir / "marginal_delta_rmse_boxplot.png",
                          config=config, dpi=dpi)
        plot_marginal_bar_means(deltas_df.copy(), key_metric,
                                outdir / "marginal_delta_rmse_bar.png",
                                feature_list=feature_list, dpi=dpi)

    plot_combo_presence_heatmap(top_df, outdir / "combo_presence_heatmap_topN.png",
                                feature_list=feature_list, config=config,
                                key_metric=key_metric, dpi=dpi)

    # Console summary
    best = summary.iloc[0]
    log(f"[BEST] {best['label']} - {key_metric.upper()}={best[key_metric]:.4f}")
    if not deltas_df.empty:
        g = deltas_df.groupby("feature")["delta"]
        log("[MARGINAL] Mean delta (with-without) by feature (lower is better):")
        for f in feature_list:
            if f in g.groups:
                x = g.get_group(f).dropna().values
                mu = float(np.mean(x))
                lo, hi = ci95(x)
                log(f"  {f:<16}  mean={mu:+.4f}  95%CI=({lo:+.4f}, {hi:+.4f})  n={x.size}")
            else:
                log(f"  {f:<16}  (no pairs)")

    log(f"[INFO] Figures written to: {outdir.resolve()}")
    log("[INFO] Done.")
    return outdir


def main():
    pa = argparse.ArgumentParser(description="Analyze feature sweep results.")
    pa.add_argument("--results-dir", type=Path, required=True)
    pa.add_argument(
        "--feature-type", type=str, required=True, choices=["wind", "aux"],
    )
    pa.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae"])
    pa.add_argument("--stationwise", action="store_true")
    pa.add_argument("--top-k", type=int, default=12)
    pa.add_argument("--dpi", type=int, default=300)
    args = pa.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(args.results_dir)

    if args.feature_type == "wind":
        feature_list = list(WIND_FEATURES)
        config = WIND_PLOT_CONFIG
    else:
        feature_list = list(AUX_GROUPS)
        config = AUX_PLOT_CONFIG

    analyze_sweep_results(
        results_dir=args.results_dir,
        feature_list=feature_list,
        config=config,
        key_metric=args.metric,
        stationwise=args.stationwise,
        top_k=args.top_k,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
