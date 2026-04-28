"""Interannual variability of wind speeds at GS sites.

Per-site box-and-whisker plot of annual mean wind speeds, sorted by
long-term mean, with an inset histogram of per-site ranges.

Supports single-panel (observations) and multi-panel (dataset comparison)
modes.

Inputs
------
--infile :
    Hourly wind-speed table (CSV or pickle) with at least:
    site_id, height, datetime_rounded (or datetime), and one or more
    wind-speed columns.

Outputs
-------
Single mode:
  interannual_variability.{png,pdf}

Dataset comparison mode (--datasets):
  dataset_interannual_variability.{png,pdf}

Usage:
  wem-interannual --infile data.pkl --outdir out
  wem-interannual --infile data.pkl --outdir out \\
      --datasets ws_modeled_wtk:WTK:#F28E2B ws_modeled_era5:ERA5:#4E79A7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from wem.utils.logging import log


# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------

def _pick_datetime_col(cols: list[str], preferred: list[str]) -> str:
    for c in preferred:
        if c in cols:
            return c
    raise ValueError(
        f"None of the datetime columns {preferred} found. Have: {list(cols)}"
    )


def load_timeseries(path: Path, site_col: str, obs_col: str) -> pd.DataFrame:
    """Load CSV or pickle, return DataFrame with required columns."""
    log(f"Loading {path.name} ...")
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
    for c in [site_col, obs_col]:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in {path.name}")
    return df


def compute_interannual_variability(
    df: pd.DataFrame,
    site_col: str = "site_height",
    obs_col: str = "ws_observed",
    datetime_cols_priority: list[str] | None = None,
    min_years_per_site: int = 3,
    completeness_frac: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-site, per-year mean wind speed with quality filtering.

    Returns
    -------
    per_site :
        One row per site with n_years, min_norm, max_norm, range_norm.
    per_year_norm :
        Per-site per-year means (filtered), suitable for plotting.
    """
    if datetime_cols_priority is None:
        datetime_cols_priority = ["datetime_rounded", "datetime"]

    dt_col = _pick_datetime_col(df.columns, datetime_cols_priority)
    work = df[[site_col, dt_col, obs_col]].copy()

    work[obs_col] = pd.to_numeric(work[obs_col], errors="coerce")
    work[dt_col] = pd.to_datetime(work[dt_col], utc=True, errors="coerce")
    work = work.dropna(subset=[site_col, obs_col, dt_col])
    if work.empty:
        raise ValueError("No valid rows after coercion.")

    work["year"] = work[dt_col].dt.year

    # Per site-year: sample count & mean
    g = work.groupby([site_col, "year"], observed=True)
    sy = g.agg(
        n_samples=(obs_col, "size"), mean_ws=(obs_col, "mean")
    ).reset_index()

    # Keep years with >= completeness_frac of the site's median yearly count
    med_counts = (
        sy.groupby(site_col, observed=True)["n_samples"]
        .median()
        .rename("median_yearly_n")
    )
    sy = sy.merge(med_counts, on=site_col, how="left")
    sy["keep_year"] = sy["n_samples"] >= (completeness_frac * sy["median_yearly_n"])
    sy_kept = sy[sy["keep_year"]].copy()

    # Drop sites with too few valid years
    years_per_site = (
        sy_kept.groupby(site_col, observed=True)["year"]
        .nunique()
        .rename("n_years")
    )
    good_sites = years_per_site[years_per_site >= min_years_per_site].index
    sy_kept = sy_kept[sy_kept[site_col].isin(good_sites)].copy()
    if sy_kept.empty:
        raise ValueError("No sites left after filtering.")

    # Normalize per-site per-year means by site median
    site_medians = (
        sy_kept.groupby(site_col, observed=True)["mean_ws"]
        .median()
        .rename("site_median_mean")
    )
    sy_kept = sy_kept.merge(site_medians, on=site_col, how="left")
    sy_kept = sy_kept[sy_kept["site_median_mean"] > 0].copy()
    sy_kept["norm_mean"] = sy_kept["mean_ws"] / sy_kept["site_median_mean"]

    # Per-site range of normalized means
    per_site = (
        sy_kept.groupby(site_col, observed=True)["norm_mean"]
        .agg(min_norm="min", max_norm="max")
        .reset_index()
    )
    per_site["range_norm"] = per_site["max_norm"] - per_site["min_norm"]
    per_site = per_site.merge(
        years_per_site.rename("n_years").reset_index(), on=site_col, how="left"
    )

    per_year_norm = sy_kept[
        [site_col, "year", "n_samples", "mean_ws", "site_median_mean", "norm_mean"]
    ].copy()

    return per_site, per_year_norm


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

def _compute_panel_stats(
    per_year_norm: pd.DataFrame,
    site_col: str,
    y_col: str,
    order: str,
) -> pd.DataFrame:
    """Aggregate per-site stats for a single panel."""
    df = per_year_norm[[site_col, "year", y_col]].dropna().copy()

    grp = df.groupby(site_col, observed=True)[y_col]
    stats = pd.DataFrame({
        "ymin": grp.min(),
        "median": grp.median(),
        "ymax": grp.max(),
    }).reset_index()
    q1 = grp.quantile(0.25).rename("q1").reset_index()
    q3 = grp.quantile(0.75).rename("q3").reset_index()
    stats = stats.merge(q1, on=site_col).merge(q3, on=site_col)

    ny = (
        df.groupby(site_col, observed=True)["year"]
        .nunique()
        .rename("n_years")
        .reset_index()
    )
    stats = stats.merge(ny, on=site_col)

    if order == "median_desc":
        stats = stats.sort_values("median", ascending=False, kind="mergesort")
    elif order == "median_asc":
        stats = stats.sort_values("median", ascending=True, kind="mergesort")
    else:
        stats = stats.sort_values(site_col, kind="mergesort")

    stats = stats.reset_index(drop=True)
    stats["site_index"] = np.arange(len(stats))
    return stats


def _draw_panel(
    ax: plt.Axes,
    stats: pd.DataFrame,
    per_year_norm: pd.DataFrame,
    site_col: str,
    y_col: str,
    color: str,
    bin_width: float,
    show_ylabel: bool = True,
) -> None:
    """Draw box-and-whisker + inset histogram onto an existing Axes."""
    # Range lines
    ax.vlines(
        stats["site_index"], stats["ymin"], stats["ymax"],
        colors="#666666", linewidth=0.8, alpha=0.9, zorder=1,
    )

    # IQR rectangles
    box_half_width = 0.35
    for _, r in stats.iterrows():
        ax.add_patch(plt.Rectangle(
            (r["site_index"] - box_half_width, r["q1"]),
            width=2 * box_half_width,
            height=max(0.0, r["q3"] - r["q1"]),
            facecolor=color, edgecolor=color,
            linewidth=0.8, alpha=0.35, zorder=2,
        ))

    # Median tick
    ax.hlines(
        stats["median"],
        stats["site_index"] - 0.32, stats["site_index"] + 0.32,
        colors="#111111", linewidth=1.4, zorder=3,
    )

    ax.set_xlim(-0.75, len(stats) - 0.25)
    ax.set_xlabel("Site index (0 \u2026 N\u22121)", fontsize=18)
    if show_ylabel:
        ax.set_ylabel(r"Average wind speed ($m\,s^{-1}$)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    if len(stats) <= 40:
        ax.set_xticks(stats["site_index"])
    else:
        step = 20
        ax.set_xticks(np.arange(0, len(stats), step))

    # Inset histogram
    df = per_year_norm[[site_col, "year", y_col]].dropna()
    ranges = (
        df.groupby(site_col, observed=True)[y_col]
        .agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
        .astype(float)
    )
    x = ranges.to_numpy()
    x = x[np.isfinite(x)]

    if x.size > 0:
        ax_hist = inset_axes(
            ax, width="30%", height="38%", loc="upper right", borderpad=0.8,
        )
        bins = np.arange(0.0, 3.0 + bin_width * 0.5, bin_width)
        ax_hist.hist(x, bins=bins, edgecolor="#333333", alpha=0.9, color=color)
        ax_hist.set_xlim(0.0, 3.0)
        ax_hist.set_xlabel(r"Range ($m\,s^{-1}$)", fontsize=14)
        ax_hist.set_ylabel("", fontsize=12)
        ax_hist.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        ax_hist.tick_params(axis="x", labelsize=13)
        ax_hist.tick_params(axis="y", labelsize=11)


# -------------------------------------------------------------------
# Public plotting functions
# -------------------------------------------------------------------

def plot_interannual(
    per_year_norm: pd.DataFrame,
    out_path: Path,
    site_col: str = "site_height",
    y_col: str = "mean_ws",
    color: str = "#AAAAAA",
    order: str = "median_desc",
    bin_width: float = 0.20,
    save_index_map: bool = True,
    index_map_path: Path | None = None,
) -> None:
    """Single-panel interannual range plot with inset histogram."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _compute_panel_stats(per_year_norm, site_col, y_col, order)
    if stats.empty:
        log("[WARN] No per-site stats to plot.")
        return

    if save_index_map:
        if index_map_path is None:
            index_map_path = out_path.with_name("interannual_site_index_map.csv")
        stats[["site_index", site_col, "n_years"]].to_csv(
            index_map_path, index=False
        )
        log(f"Wrote site index map -> {index_map_path.name}")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300, layout="constrained")
    _draw_panel(ax, stats, per_year_norm, site_col, y_col, color, bin_width)

    log(f"Rendering interannual variability plot -> {out_path.name}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_interannual_compare(
    panels: list[tuple[str, str, pd.DataFrame]],
    out_path: Path,
    site_col: str = "site_height",
    y_col: str = "mean_ws",
    order: str = "median_desc",
    bin_width: float = 0.20,
) -> None:
    """Multi-panel interannual comparison figure.

    Parameters
    ----------
    panels :
        List of (label, color, per_year_norm) tuples, one per panel.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)

    fig, axes = plt.subplots(
        1, n, figsize=(10 * n, 5.5), dpi=300, layout="constrained",
    )
    if n == 1:
        axes = [axes]

    for i, (label, color, pyn) in enumerate(panels):
        ax = axes[i]
        stats = _compute_panel_stats(pyn, site_col, y_col, order)
        if stats.empty:
            log(f"[WARN] No data for {label}; skipping panel.")
            continue
        _draw_panel(
            ax, stats, pyn, site_col, y_col, color, bin_width,
            show_ylabel=(i == 0),
        )
        panel_letter = chr(ord("a") + i)
        ax.annotate(
            f"({panel_letter}) {label}",
            xy=(0.5, 0), xycoords="axes fraction",
            xytext=(0, -52), textcoords="offset points",
            ha="center", va="top", fontsize=18,
        )

    log(f"Rendering dataset interannual variability -> {out_path.name}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def _parse_dataset_spec(spec: str) -> tuple[str, str, str]:
    """Parse 'column:Label:color' dataset spec."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected col:Label:#color, got '{spec}'"
        )
    return parts[0], parts[1], parts[2]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot interannual variability of wind speeds at GS sites.",
    )
    ap.add_argument(
        "--infile", type=Path, required=True,
        help="Hourly wind-speed table (CSV or .pkl) with site_id, height, "
             "and a datetime column.",
    )
    ap.add_argument("--outdir", type=Path, default=Path("analysis_out"))
    ap.add_argument(
        "--site-col", default="site_id",
        help="Column identifying unique sites (default: site_id).",
    )
    ap.add_argument(
        "--height-col", default="height",
        help="Column for measurement height (default: height).",
    )
    ap.add_argument(
        "--obs-col", default="ws_observed",
        help="Column for observed wind speed; used in single-panel mode "
             "(default: ws_observed).",
    )
    ap.add_argument(
        "--datasets", nargs="+", metavar="COL:LABEL:COLOR",
        help="Multi-panel mode: one or more col:Label:#color specs. "
             "Example: ws_modeled_wtk:WTK:#F28E2B ws_modeled_era5:ERA5:#4E79A7",
    )
    ap.add_argument(
        "--dt-col", default=None,
        help="Force a datetime column; default tries datetime_rounded then datetime.",
    )
    ap.add_argument("--min-years", type=int, default=3)
    ap.add_argument("--completeness", type=float, default=0.95)
    ap.add_argument("--bin-width", type=float, default=0.20)
    ap.add_argument(
        "--color", default="#AAAAAA",
        help="Box/histogram fill color for single-panel mode (default: #AAAAAA).",
    )
    ap.add_argument(
        "--format", choices=["png", "pdf"], default="png",
        help="Output image format (default: png).",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    dt_priority = (
        [args.dt_col] if args.dt_col
        else ["datetime_rounded", "datetime"]
    )

    # Determine which columns we need to validate on load
    if args.datasets:
        dataset_specs = [_parse_dataset_spec(s) for s in args.datasets]
        check_col = dataset_specs[0][0]
    else:
        dataset_specs = None
        check_col = args.obs_col

    # Load data
    df = load_timeseries(args.infile, args.site_col, check_col)

    # Build composite site key: site_id + height
    if args.height_col in df.columns:
        df["site_height"] = (
            df[args.site_col].astype(str) + df[args.height_col].astype(str)
        )
        site_key = "site_height"
    else:
        site_key = args.site_col

    if dataset_specs:
        # Multi-panel mode
        panels = []
        for col, label, color in dataset_specs:
            if col not in df.columns:
                log(f"[WARN] Column '{col}' not found; skipping {label}.")
                continue
            log(f"Computing interannual variability for {label} ...")
            _, pyn = compute_interannual_variability(
                df, site_col=site_key, obs_col=col,
                datetime_cols_priority=dt_priority,
                min_years_per_site=args.min_years,
                completeness_frac=args.completeness,
            )
            panels.append((label, color, pyn))

        out_path = args.outdir / f"dataset_interannual_variability.{args.format}"
        plot_interannual_compare(
            panels, out_path=out_path, site_col=site_key,
            bin_width=args.bin_width,
        )
    else:
        # Single-panel mode
        log("Computing interannual variability ...")
        per_site, per_year_norm = compute_interannual_variability(
            df, site_col=site_key, obs_col=args.obs_col,
            datetime_cols_priority=dt_priority,
            min_years_per_site=args.min_years,
            completeness_frac=args.completeness,
        )
        log(f"Sites retained: {per_site[site_key].nunique()}")

        out_path = args.outdir / f"interannual_variability.{args.format}"
        plot_interannual(
            per_year_norm, out_path=out_path, site_col=site_key,
            y_col="mean_ws", color=args.color, bin_width=args.bin_width,
        )

    log(f"Done. Output in {args.outdir}")


if __name__ == "__main__":
    main()
