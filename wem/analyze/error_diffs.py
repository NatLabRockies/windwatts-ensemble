"""Per-station delta absolute bias bar charts, colored by RdBu gradient.

For each (station_id, height_m), compute:
  diff = |mean(pred) - mean(obs)| - |mean(ds) - mean(obs)|

Interpretation:
  diff < 0  -> ML has LOWER absolute bias than dataset (better)  -> BLUE
  diff > 0  -> ML has HIGHER absolute bias than dataset (worse)  -> RED

Bars are sorted from most negative to most positive. The x-axis is normalized
to [0, 1] representing the cumulative proportion of site-heights.

Usage:
  wem-error-diffs data.csv --outdir figs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from wem.constants import DATASET_LABELS
from wem.utils.logging import log


DATASET_COLS_DEFAULT = [
    "era5",
    "hrrr",
    "wtk",
    "wtk_led_conus",
    "wtk_led_climate",
    "gwa_interp",
]


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce columns to numeric (errors -> NaN)."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_group_diffs(
    df: pd.DataFrame,
    dataset_col: str,
    group_cols: Tuple[str, str] = ("station_id", "height_m"),
    obs_col: str = "observation",
    ml_col: str = "pred_observation",
) -> pd.DataFrame:
    """Compute per-group delta absolute bias (ML vs dataset).

    Parameters
    ----------
    df : DataFrame
        Quantile table with observation, ML, and dataset columns.
    dataset_col : str
        Column name for the baseline dataset.
    group_cols : tuple of str
        Columns to group by (default: station_id, height_m).
    obs_col : str
        Column name for ground-truth observations.
    ml_col : str
        Column name for ML predictions.

    Returns
    -------
    DataFrame
        One row per group with ml_abs_err_mean, ds_abs_err_mean, diff.
        Sorted from most negative (ML better) to most positive (ML worse).
    """
    need = [obs_col, ml_col, dataset_col] + list(group_cols)
    df = df.dropna(subset=need).copy()

    grouped = (
        df.groupby(list(group_cols), dropna=False)[
            [ml_col, dataset_col, obs_col]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped["ml_abs_err_mean"] = (
        grouped[ml_col] - grouped[obs_col]
    ).abs()
    grouped["ds_abs_err_mean"] = (
        grouped[dataset_col] - grouped[obs_col]
    ).abs()
    grouped["diff"] = (
        grouped["ml_abs_err_mean"] - grouped["ds_abs_err_mean"]
    )
    grouped = grouped.sort_values("diff", ascending=True).reset_index(
        drop=True
    )
    return grouped


def make_bar_chart(
    diffs_df: pd.DataFrame,
    dataset_label: str,
    outpath: str | None = None,
    figsize: tuple = (10, 5),
    clim: tuple = (-2.5, 2.5),
    ylim: tuple | None = None,
) -> None:
    """Plot vertical bars colored by RdBu gradient."""
    diffs = diffs_df["diff"].to_numpy(dtype=float)
    n = len(diffs)
    if n == 0:
        log(f"[WARN] No rows to plot for {dataset_label}.")
        return

    width = 1.0 / n
    x_left = np.arange(n) / n
    x_centers = x_left + width / 2.0

    vmin, vmax = clim
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.colormaps["RdBu_r"]
    colors = cmap(norm(diffs))

    frac_ml_better = float((diffs < 0).sum()) / n

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x_centers,
        diffs,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.25,
    )

    if ylim is None:
        ax.set_ylim([vmin, vmax])
    else:
        ax.set_ylim(ylim)

    ax.axhline(0, linewidth=1.0, color="black")
    ax.axvline(
        frac_ml_better, linestyle="--", linewidth=1.2, color="black"
    )

    ymin_ax, ymax_ax = ax.get_ylim()
    yspan = ymax_ax - ymin_ax
    ax.text(
        frac_ml_better,
        ymax_ax - 0.03 * yspan,
        f"{frac_ml_better:.0%} ML better",
        ha="center",
        va="top",
        fontsize=10,
        rotation=0,
        bbox=dict(
            boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85
        ),
    )

    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(
        "Proportion of site-heights (sorted by error difference)"
    )
    ax.set_ylabel(
        r"Avg $|$ML $-$ obs$|$ $-$ Avg $|$dataset $-$ obs$|$  [$m\,s^{-1}$]"
    )
    ax.set_title(
        f"{dataset_label} vs ML \u2014 Difference in Absolute Bias\n"
        "(negative: ML has lower bias; positive: ML has higher bias)"
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    if outpath and outpath != "-":
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200)
        log(f"Saved: {outpath}")
        plt.close(fig)
    else:
        plt.show()


def _plot_bar_panel(
    ax: plt.Axes,
    diffs_df: pd.DataFrame,
    dataset_label: str,
    norm: TwoSlopeNorm,
    cmap,
    clim: Tuple[float, float],
    ylim: tuple | None = None,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
) -> None:
    """Draw one bar panel into an existing Axes."""
    diffs = diffs_df["diff"].to_numpy(dtype=float)
    n = len(diffs)
    if n == 0:
        ax.set_visible(False)
        return

    width = 1.0 / n
    x_left = np.arange(n) / n
    x_centers = x_left + width / 2.0

    colors = cmap(norm(diffs))
    frac_ml_better = float((diffs < 0).sum()) / n

    ax.bar(
        x_centers,
        diffs,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.05,
    )

    vmin, vmax = clim
    if ylim is None:
        ax.set_ylim([vmin, vmax])
    else:
        ax.set_ylim(ylim)

    ax.axhline(0, linewidth=1.0, color="black")
    ax.axvline(
        frac_ml_better, linestyle="--", linewidth=1.0, color="black"
    )

    ymin_ax, ymax_ax = ax.get_ylim()
    yspan = ymax_ax - ymin_ax
    ax.text(
        frac_ml_better,
        ymax_ax - 0.05 * yspan,
        f"{frac_ml_better:.0%}",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85
        ),
    )

    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

    if show_xlabel:
        ax.set_xlabel("Proportion of site-heights", fontsize=14)
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel(r"ML Bias $-$ Dataset Bias ($m\,s^{-1}$)", fontsize=12)
        ax.set_yticks(
            np.arange(int(np.floor(vmin)), int(np.ceil(vmax)) + 1)
        )
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.text(
        0.02,
        0.96,
        dataset_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)


def make_bar_chart_grid(
    diffs_by_dataset: List[Tuple[str, pd.DataFrame]],
    outpath: str | None = None,
    figsize: tuple = (12, 6),
    clim: tuple = (-2.5, 2.5),
    ylim: tuple | None = None,
) -> None:
    """Create a 2x3 grid of bar charts, one panel per dataset."""
    vmin, vmax = clim
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.colormaps["RdBu_r"]

    desired_order = [
        "WTK-LED\nCONUS",
        "WTK-LED\nClimate",
        "ERA5",
        "HRRR",
        "WTK",
        "GWA",
    ]
    order_index = {lab: i for i, lab in enumerate(desired_order)}

    diffs_by_dataset_sorted = sorted(
        diffs_by_dataset,
        key=lambda t: order_index.get(t[0], len(desired_order)),
    )

    fig, axes = plt.subplots(2, 3, figsize=(figsize[0], figsize[1] + 2), sharey=True)
    axes_flat = axes.flatten()

    for idx, (label, df) in enumerate(diffs_by_dataset_sorted):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        row = idx // 3
        col = idx % 3
        show_xlabel = row == 1
        show_ylabel = col == 0

        _plot_bar_panel(
            ax,
            diffs_df=df,
            dataset_label=label,
            norm=norm,
            cmap=cmap,
            clim=clim,
            ylim=ylim,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
        )

    for j in range(len(diffs_by_dataset), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.5)
    if outpath and outpath != "-":
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300)
        log(f"Saved grid figure: {outpath}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot delta absolute bias bar charts by site-height "
        "(colored by RdBu)."
    )
    parser.add_argument(
        "csv",
        help="Path to CSV with columns: station_id, height_m, observation, "
        "pred_observation, and wind resource dataset columns.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_COLS_DEFAULT,
        help="Wind resource dataset columns to compare against ML.",
    )
    parser.add_argument(
        "--outdir",
        default="figs",
        help="Directory to save figures. Use '-' to show plots instead.",
    )
    parser.add_argument(
        "--station_col", default="station_id", help="Station id column."
    )
    parser.add_argument(
        "--height_col", default="height_m", help="Height column."
    )
    parser.add_argument(
        "--obs_col", default="observation", help="Observation column."
    )
    parser.add_argument(
        "--ml_col", default="pred_observation", help="ML prediction column."
    )
    parser.add_argument(
        "--clim",
        type=float,
        nargs=2,
        default=[-2.5, 2.5],
        help="Color mapping limits [vmin vmax] for RdBu.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        help="Y-axis limits [ymin ymax].",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    numeric_cols = (
        [args.obs_col, args.ml_col, args.height_col] + args.datasets
    )
    _coerce_numeric(df, numeric_cols)

    # Individual charts per dataset
    for ds in args.datasets:
        diffs_df = compute_group_diffs(
            df,
            dataset_col=ds,
            group_cols=(args.station_col, args.height_col),
            obs_col=args.obs_col,
            ml_col=args.ml_col,
        )
        ds_label = DATASET_LABELS.get(ds, ds)
        outpath = (
            None
            if args.outdir == "-"
            else str(
                Path(args.outdir) / f"{ds}_vs_ml_error_diff.png"
            )
        )
        make_bar_chart(
            diffs_df,
            dataset_label=ds_label,
            outpath=outpath,
            clim=tuple(args.clim),
            ylim=(
                tuple(args.ylim)
                if args.ylim is not None
                else None
            ),
        )

    # Grid label map with newlines for multi-line labels
    grid_label_map = {
        "era5": "ERA5",
        "hrrr": "HRRR",
        "wtk": "WTK",
        "wtk_led_conus": "WTK-LED\nCONUS",
        "wtk_led_climate": "WTK-LED\nClimate",
        "gwa_interp": "GWA",
    }

    diffs_by_dataset: List[Tuple[str, pd.DataFrame]] = []
    for ds in args.datasets:
        diffs_df = compute_group_diffs(
            df,
            dataset_col=ds,
            group_cols=(args.station_col, args.height_col),
            obs_col=args.obs_col,
            ml_col=args.ml_col,
        )
        ds_label = grid_label_map.get(ds, ds)
        diffs_by_dataset.append((ds_label, diffs_df))

    if args.outdir == "-":
        grid_outpath = None
    else:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        grid_outpath = str(
            Path(args.outdir) / "error_diff_bar_grid.png"
        )

    make_bar_chart_grid(
        diffs_by_dataset,
        outpath=grid_outpath,
        clim=tuple(args.clim),
        ylim=(
            tuple(args.ylim) if args.ylim is not None else None
        ),
    )


if __name__ == "__main__":
    main()
