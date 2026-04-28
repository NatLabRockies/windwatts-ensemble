"""Per-site quantile / CDF curves for Gold Standard sites.

Reads a CSV with quantile data and, for every GS site (station_id, height_m),
creates a separate PNG showing:

- Observation (thick blue)
- ERA5, WTK, HRRR, WTK-LED CONUS, WTK-LED Climate (light green dashed)
- Optionally: ML model prediction (thick black, with legend)

Usage:
  wem-site-cdfs --csv ml_training_data.csv --outdir gs_cdf_curves
  wem-site-cdfs --csv ml_results.csv --outdir ml_cdf --ml-col pred_observation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from wem.constants import DATASET_LABELS
from wem.utils.logging import log

# Colors
COLOR_DATASET = "#72B857"  # light green, dashed
COLOR_OBS = "#0070C0"  # blue (GS-only mode) / "#4E88DD" (ML mode)
COLOR_ML = "#000000"  # black
AXIS_COLOR = "#444444"
GRID_COLOR = "#BBBBBB"

ORDER_PRETTY = ["ERA5", "WTK", "HRRR", "WTK-LED CONUS", "WTK-LED Climate"]


def validate_columns(
    df: pd.DataFrame, required: List[str]
) -> List[str]:
    """Return list of missing columns (empty if all present)."""
    return [c for c in required if c not in df.columns]


def compute_xlim(
    df: pd.DataFrame, value_cols: List[str]
) -> tuple[float, float]:
    """Compute global x-limits based on 99th percentile of value columns."""
    all_vals = df[value_cols].to_numpy().ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        xmax = float(np.nanpercentile(all_vals, 99)) * 1.05
    else:
        xmax = 20.0
    return (0.0, xmax)


def style_cdf_axes(ax: plt.Axes, xlim: tuple[float, float]) -> None:
    """Apply shared CDF styling: spines, ticks, grid."""
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        labelbottom=False,
        labelleft=False,
        length=4,
        width=1.5,
        color=AXIS_COLOR,
    )

    for spine_name, spine in ax.spines.items():
        if spine_name in ("left", "bottom"):
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color(AXIS_COLOR)
        else:
            spine.set_visible(False)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color=GRID_COLOR,
        alpha=0.8,
    )


def plot_site_cdf(
    group: pd.DataFrame,
    xlim: tuple[float, float],
    outdir: Path,
    *,
    figsize: tuple[float, float] = (4.0, 4.0),
    obs_color: str = COLOR_OBS,
    obs_linewidth: float = 4.0,
    show_ml: bool = False,
    ml_col: str = "pred_observation",
    ml_color: str = COLOR_ML,
    ml_linewidth: float = 3.0,
    show_legend: bool = False,
    filename_suffix: str = "",
) -> None:
    """Make one CDF plot for a single (station_id, height_m) group."""
    group = group.sort_values("qnum")
    q = group["qnum"].to_numpy(dtype=float)
    y_cdf = q / 100.0

    obs = group["observation"].to_numpy()
    col_map = {
        "ERA5": group["era5"].to_numpy(),
        "WTK": group["wtk"].to_numpy(),
        "HRRR": group["hrrr"].to_numpy(),
        "WTK-LED CONUS": group["wtk_led_conus"].to_numpy(),
        "WTK-LED Climate": group["wtk_led_climate"].to_numpy(),
    }

    station_id = str(group["station_id"].iloc[0])
    name = (
        str(group["name"].iloc[0])
        if "name" in group.columns
        else station_id
    )
    height = (
        float(group["height_m"].iloc[0])
        if "height_m" in group.columns
        else np.nan
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_facecolor("white")

    # Datasets: light green dashed
    for label in ORDER_PRETTY:
        x = col_map[label]
        ax.plot(
            x,
            y_cdf,
            color=COLOR_DATASET,
            linewidth=1.8,
            linestyle="--",
            zorder=3,
            solid_joinstyle="round",
        )

    # Observations
    ax.plot(
        obs,
        y_cdf,
        color=obs_color,
        linewidth=obs_linewidth,
        zorder=4,
        solid_joinstyle="round",
    )

    # ML predictions (optional)
    if show_ml and ml_col in group.columns:
        ml_pred = group[ml_col].to_numpy()
        ax.plot(
            ml_pred,
            y_cdf,
            color=ml_color,
            linewidth=ml_linewidth,
            zorder=5,
            solid_joinstyle="round",
        )

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                color=ml_color,
                linewidth=3.0,
                label="ML model",
                solid_joinstyle="round",
            ),
            Line2D(
                [0],
                [0],
                color=obs_color,
                linewidth=5,
                label="Observations",
                solid_joinstyle="round",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_DATASET,
                linewidth=1.8,
                linestyle="--",
                label="Baseline datasets",
                solid_joinstyle="round",
            ),
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=15,
            loc="lower right",
            handlelength=2.0,
        )

    style_cdf_axes(ax, xlim)
    fig.tight_layout()

    safe_name = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in name
    )
    fname = f"{station_id}_{safe_name}_h{height:.0f}m{filename_suffix}.png"
    out_path = outdir / fname
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Make per-site CDF plots from quantile data."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV with quantiles.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("cdf_curves"),
        help="Directory to save per-site images.",
    )
    parser.add_argument(
        "--ml-col",
        type=str,
        default=None,
        help="If set, plot ML predictions from this column; "
        "triggers legend and adjusts styling.",
    )
    parser.add_argument(
        "--gs-col",
        type=str,
        default="observation_type",
        help="Column to filter GS sites.",
    )
    parser.add_argument(
        "--gs-value",
        type=str,
        default="GS",
        help="Value identifying GS rows (use '1' with --gs-col is_gs).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Filter to GS sites
    if args.gs_col in df.columns:
        col_vals = df[args.gs_col]
        # Try numeric comparison first (for is_gs=1)
        try:
            gs_val_num = float(args.gs_value)
            df = df[
                pd.to_numeric(col_vals, errors="coerce") == gs_val_num
            ].copy()
        except ValueError:
            df = df[col_vals == args.gs_value].copy()

    base_needed = [
        "station_id",
        "qnum",
        "observation",
        "era5",
        "hrrr",
        "wtk",
        "wtk_led_conus",
        "wtk_led_climate",
    ]
    needed = list(base_needed)
    if args.ml_col:
        needed.append(args.ml_col)

    missing = validate_columns(df, needed)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=needed)

    if "height_m" not in df.columns:
        df["height_m"] = 10.0

    # Compute global x-limits
    value_cols = [
        "observation",
        "era5",
        "wtk",
        "hrrr",
        "wtk_led_conus",
        "wtk_led_climate",
    ]
    if args.ml_col:
        value_cols.append(args.ml_col)
    xlim = compute_xlim(df, value_cols)

    args.outdir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["station_id", "height_m"])

    show_ml = args.ml_col is not None
    log(
        f"Found {len(grouped)} site/height groups; "
        f"writing figures to {args.outdir}..."
    )

    # Style adjustments for ML mode
    figsize = (4.0, 3.0) if show_ml else (4.0, 4.0)
    obs_color = "#4E88DD" if show_ml else COLOR_OBS
    obs_linewidth = 8.0 if show_ml else 4.0
    suffix = "_ml_results" if show_ml else ""

    for (_, _), g in grouped:
        plot_site_cdf(
            g,
            xlim,
            args.outdir,
            figsize=figsize,
            obs_color=obs_color,
            obs_linewidth=obs_linewidth,
            show_ml=show_ml,
            ml_col=args.ml_col or "pred_observation",
            show_legend=show_ml,
            filename_suffix=suffix,
        )

    log("Done.")


if __name__ == "__main__":
    main()
