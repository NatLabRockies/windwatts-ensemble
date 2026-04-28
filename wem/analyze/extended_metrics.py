"""Compute summary bias metrics for wind resource datasets.

Compute summary bias metrics (median bias, mean |bias|, median |bias|, and
mean/median absolute percentage bias) for each wind resource dataset and for
each observational subset (ASOS vs GS), suitable for filling
Table~\\ref{tab:dataset-summary} in the paper and for additional
normalized-error diagnostics.

Inputs
------
--infile :
    Long-format quantile table (CSV or Parquet).
--gwa (optional) :
    CSV with per-site/height Global Wind Atlas means.

Outputs
-------
1) dataset_summary.csv
2) dataset_summary_by_height_gs.csv
3) height_dependence_gs.png
4) height_bias_boxplots_gs.png
5) LaTeX table rows printed to stdout

Usage:
  wem-analyze-extended --infile combined_quantiles_long_with_topo_loocv_10km.csv \\
      --gwa site_height_ws_avg_with_gwa.csv --outdir analysis_out_table
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from wem.utils.quantiles import mean_from_quantile_long as mean_from_quantile_series
from wem.constants import DATASET_LABELS, DATASET_LATEX
from wem.utils.logging import log
from wem.utils.sites import normalize_obs_type


# ---------------------------------------------------------------------------
# Aggregation from long-format quantiles -> site/height means & biases
# ---------------------------------------------------------------------------


def aggregate_site_means(
    df: pd.DataFrame,
    dataset_cols: List[str],
    min_qrows: int = 10,
) -> pd.DataFrame:
    """Aggregate a long-format quantile table to site/height-level means.

    Parameters
    ----------
    df : DataFrame
        Long-format table with at least columns:
        station_id, observation_type, height_m, qnum, observation,
        and dataset_cols.
    dataset_cols : list of str
        Column names for wind resource datasets.
    min_qrows : int
        Minimum number of finite quantile rows required per group.

    Returns
    -------
    site : DataFrame
        One row per (station_id, height_m), with mean_observation,
        mean_<dataset>, bias_<dataset>, absbias_<dataset>.
    """
    needed = [
        "station_id",
        "observation_type",
        "height_m",
        "qnum",
        "observation",
    ]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Input missing required column: {c!r}")

    df = df.copy()
    df["qnum"] = pd.to_numeric(df["qnum"], errors="coerce")

    dataset_cols = [c for c in dataset_cols if c in df.columns]
    if not dataset_cols:
        raise ValueError(
            "No requested dataset columns found in input file."
        )

    rows: List[Dict] = []

    for (sid, h), g in df.groupby(
        ["station_id", "height_m"], sort=False
    ):
        if np.isfinite(g["qnum"]).sum() < min_qrows:
            continue

        g = g.sort_values("qnum")
        row: Dict = {
            "station_id": str(sid),
            "height_m": float(g["height_m"].iloc[0]),
            "observation_type": normalize_obs_type(
                str(g["observation_type"].iloc[0])
            ),
        }

        obs_mean = mean_from_quantile_series(
            g["qnum"].to_numpy(),
            pd.to_numeric(g["observation"], errors="coerce").to_numpy(),
        )
        row["mean_observation"] = obs_mean

        for d in dataset_cols:
            vals = pd.to_numeric(g[d], errors="coerce").to_numpy()
            d_mean = mean_from_quantile_series(
                g["qnum"].to_numpy(), vals
            )
            row[f"mean_{d}"] = d_mean
            if np.isfinite(d_mean) and np.isfinite(obs_mean):
                b = d_mean - obs_mean
                row[f"bias_{d}"] = b
                row[f"absbias_{d}"] = abs(b)
            else:
                row[f"bias_{d}"] = np.nan
                row[f"absbias_{d}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def merge_gwa_means(
    site: pd.DataFrame, gwa_path: Path
) -> pd.DataFrame:
    """Merge GWA per-site/height mean wind speeds and compute biases."""
    log(f"[INFO] Merging GWA means from {gwa_path} ...")
    gwa = pd.read_csv(
        gwa_path, dtype={"station_id": str}, low_memory=False
    )

    if (
        "station_id" not in gwa.columns
        or "height_m" not in gwa.columns
    ):
        raise ValueError(
            "GWA CSV must contain 'station_id' and 'height_m' columns."
        )

    gwa_col = None
    for cand in ["gwa_interp", "gwa", "mean_gwa"]:
        if cand in gwa.columns:
            gwa_col = cand
            break
    if gwa_col is None:
        raise ValueError(
            "GWA CSV must contain a mean-wind column "
            "(e.g., 'gwa_interp')."
        )

    gwa = gwa[["station_id", "height_m", gwa_col]].copy()
    gwa["height_m"] = pd.to_numeric(gwa["height_m"], errors="coerce")
    gwa = gwa.dropna(
        subset=["station_id", "height_m"]
    ).drop_duplicates(subset=["station_id", "height_m"])

    site = site.merge(
        gwa.rename(columns={gwa_col: "mean_gwa"}),
        on=["station_id", "height_m"],
        how="left",
    )

    if (
        "mean_gwa" in site.columns
        and "mean_observation" in site.columns
    ):
        b = site["mean_gwa"] - site["mean_observation"]
        site["bias_gwa"] = b
        site["absbias_gwa"] = np.abs(b)
        log(
            f"[INFO] GWA merged for "
            f"{int(site['mean_gwa'].notna().sum()):,} site-height rows."
        )
    else:
        log(
            "[WARN] Could not compute GWA bias "
            "(missing mean_gwa or mean_observation)."
        )

    return site


# ---------------------------------------------------------------------------
# Summary metrics for ASOS / GS
# ---------------------------------------------------------------------------


def compute_summary_metrics(
    site: pd.DataFrame,
    dataset_keys: List[str],
) -> pd.DataFrame:
    """Compute summary metrics for each dataset and subset (ASOS, GS).

    Metrics: median_bias, mean_abs_bias, median_abs_bias,
    mean_abs_pct_bias, median_abs_pct_bias, n_sites.
    """
    site = site.copy()
    site["observation_type"] = (
        site["observation_type"].astype(str).map(normalize_obs_type)
    )

    subsets = ["ASOS", "GS"]
    records: List[Dict] = []

    for subset in subsets:
        sub = site[site["observation_type"] == subset].copy()
        if sub.empty:
            log(
                f"[WARN] No rows found for subset={subset}. Skipping."
            )
            continue

        obs_means_all = pd.to_numeric(
            sub["mean_observation"], errors="coerce"
        ).to_numpy(dtype="float64")

        for d in dataset_keys:
            bcol = f"bias_{d}"
            if bcol not in sub.columns:
                continue

            arr_bias_all = pd.to_numeric(
                sub[bcol], errors="coerce"
            ).to_numpy(dtype="float64")

            mask = (
                np.isfinite(arr_bias_all)
                & np.isfinite(obs_means_all)
                & (obs_means_all > 0.0)
            )
            if not np.any(mask):
                continue

            arr_bias = arr_bias_all[mask]
            arr_obs = obs_means_all[mask]
            arr_abs = np.abs(arr_bias)
            arr_abs_pct = arr_abs / arr_obs * 100.0

            rec = {
                "dataset": d,
                "subset": subset,
                "median_bias": float(np.nanmedian(arr_bias)),
                "mean_abs_bias": float(np.nanmean(arr_abs)),
                "median_abs_bias": float(np.nanmedian(arr_abs)),
                "mean_abs_pct_bias": float(np.nanmean(arr_abs_pct)),
                "median_abs_pct_bias": float(
                    np.nanmedian(arr_abs_pct)
                ),
                "n_sites": int(arr_bias.size),
            }
            records.append(rec)

    return pd.DataFrame(records)


def compute_height_metrics(
    site: pd.DataFrame,
    dataset_keys: List[str],
    subset_filter: str = "GS",
    round_to: int = 10,
) -> pd.DataFrame:
    """Compute bias metrics stratified by rounded height band.

    Parameters
    ----------
    site : DataFrame
        Site/height-level metrics.
    dataset_keys : list of str
        Dataset identifiers.
    subset_filter : str
        Which observation_type to keep.
    round_to : int
        Height rounding in meters.

    Returns
    -------
    height_summary : DataFrame
        Columns: dataset, subset, height_band_m, mean_bias,
        mean_abs_bias, median_abs_bias, mean_abs_pct_bias, n_sites.
    """
    site = site.copy()
    site["observation_type"] = (
        site["observation_type"].astype(str).map(normalize_obs_type)
    )

    sub = site[site["observation_type"] == subset_filter].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "subset",
                "height_band_m",
                "mean_bias",
                "mean_abs_bias",
                "median_abs_bias",
                "mean_abs_pct_bias",
                "n_sites",
            ]
        )

    sub["height_band_m"] = (
        pd.to_numeric(sub["height_m"], errors="coerce")
        / float(round_to)
    ).round() * float(round_to)

    records: List[Dict] = []

    for h_band, g_h in sub.groupby("height_band_m"):
        if not np.isfinite(h_band):
            continue

        obs = pd.to_numeric(
            g_h["mean_observation"], errors="coerce"
        ).to_numpy(dtype="float64")

        for d in dataset_keys:
            bcol = f"bias_{d}"
            if bcol not in g_h.columns:
                continue

            bias_arr = pd.to_numeric(
                g_h[bcol], errors="coerce"
            ).to_numpy(dtype="float64")

            good = (
                np.isfinite(bias_arr)
                & np.isfinite(obs)
                & (obs > 0)
            )
            if not np.any(good):
                continue

            b = bias_arr[good]
            absb = np.abs(b)
            pct = (absb / obs[good]) * 100.0

            rec = {
                "dataset": d,
                "subset": subset_filter,
                "height_band_m": float(h_band),
                "mean_bias": float(np.nanmean(b)),
                "mean_abs_bias": float(np.nanmean(absb)),
                "median_abs_bias": float(np.nanmedian(absb)),
                "mean_abs_pct_bias": float(np.nanmean(pct)),
                "n_sites": int(b.size),
            }
            records.append(rec)

    return pd.DataFrame(records)


def compute_height_bias_distributions(
    site: pd.DataFrame,
    dataset_keys: List[str],
    subset_filter: str = "GS",
    round_to: int = 10,
) -> pd.DataFrame:
    """Prepare per-site biases grouped by height band for plots.

    Returns a long-format DataFrame with columns:
      dataset, subset, height_band_m, height_m, bias.

    Uses round() for height banding (consistent with
    compute_height_metrics).
    """
    site = site.copy()
    site["observation_type"] = (
        site["observation_type"].astype(str).map(normalize_obs_type)
    )

    sub = site[site["observation_type"] == subset_filter].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "subset",
                "height_band_m",
                "height_m",
                "bias",
            ]
        )

    h = pd.to_numeric(sub["height_m"], errors="coerce")
    sub["height_band_m"] = (h / float(round_to)).round() * float(
        round_to
    )

    obs = pd.to_numeric(
        sub["mean_observation"], errors="coerce"
    ).to_numpy(dtype="float64")
    hband = pd.to_numeric(
        sub["height_band_m"], errors="coerce"
    ).to_numpy(dtype="float64")

    records = []
    for d in dataset_keys:
        bcol = f"bias_{d}"
        if bcol not in sub.columns:
            continue

        b = pd.to_numeric(sub[bcol], errors="coerce").to_numpy(
            dtype="float64"
        )

        good = (
            np.isfinite(b)
            & np.isfinite(obs)
            & (obs > 0)
            & np.isfinite(hband)
        )
        if not np.any(good):
            continue

        rec = pd.DataFrame(
            {
                "dataset": d,
                "subset": subset_filter,
                "height_band_m": hband[good],
                "height_m": h.to_numpy()[good],
                "bias": b[good],
            }
        )
        records.append(rec)

    if not records:
        return pd.DataFrame(
            columns=[
                "dataset",
                "subset",
                "height_band_m",
                "height_m",
                "bias",
            ]
        )

    return pd.concat(records, ignore_index=True)


def plot_height_dependence(
    height_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot GS height-stratified error metrics (2 panels)."""
    import matplotlib.pyplot as plt

    if height_summary is None or height_summary.empty:
        log(
            "[WARN] Height summary is empty; "
            "skipping height-dependence plot."
        )
        return

    df = height_summary.copy()
    df["subset"] = df["subset"].astype(str)
    df = df[df["subset"] == "GS"]
    if df.empty:
        log(
            "[WARN] No GS rows in height summary; "
            "skipping height-dependence plot."
        )
        return

    df["height_band_m"] = pd.to_numeric(
        df["height_band_m"], errors="coerce"
    )
    df = df[np.isfinite(df["height_band_m"])].sort_values(
        "height_band_m"
    )

    color_map = {
        "era5": "#1f77b4",
        "wtk": "#ff7f0e",
        "wtk_led_climate": "#2ca02c",
        "wtk_led_conus": "#d62728",
        "hrrr": "#9467bd",
        "bchrrr": "#8c564b",
        "gwa": "#17becf",
    }

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), sharex=True, dpi=150
    )

    datasets_in_df = sorted(df["dataset"].unique())

    ax0 = axes[0]
    for dkey in datasets_in_df:
        g = df[df["dataset"] == dkey]
        if g.empty:
            continue
        x = g["height_band_m"].values
        y = g["mean_bias"].values
        ax0.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            linewidth=1.4,
            markersize=4.5,
            label=DATASET_LABELS.get(dkey, dkey),
            color=color_map.get(dkey, None),
        )
    ax0.set_xlabel("Height band (m)")
    ax0.set_ylabel(r"Mean bias (m s$^{-1}$)")
    ax0.set_title("GS sites: mean bias vs. height")
    ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax1 = axes[1]
    for dkey in datasets_in_df:
        g = df[df["dataset"] == dkey]
        if g.empty:
            continue
        x = g["height_band_m"].values
        y = g["mean_abs_pct_bias"].values
        ax1.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            linewidth=1.4,
            markersize=4.5,
            label=DATASET_LABELS.get(dkey, dkey),
            color=color_map.get(dkey, None),
        )
    ax1.set_xlabel("Height band (m)")
    ax1.set_ylabel(r"Mean $|\mathrm{bias}|$ / mean WS (\%)")
    ax1.set_title(
        "GS sites: mean absolute percentage bias vs. height"
    )
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] Wrote height-dependence figure -> {out_path}")


def plot_height_bias_raw_panels(
    height_bias: pd.DataFrame,
    out_path: Path,
    subset_filter: str = "GS",
) -> None:
    """Plot raw (height_m, bias) scatter in 3x2 panels per dataset."""
    import matplotlib.pyplot as plt

    if height_bias is None or height_bias.empty:
        log(
            "[WARN] Height-bias table is empty; "
            "skipping raw-panel figure."
        )
        return

    df = height_bias.copy()
    df["subset"] = df["subset"].astype(str)
    df = df[df["subset"] == subset_filter]
    if df.empty:
        log(
            f"[WARN] No rows for subset={subset_filter}; "
            "skipping raw-panel figure."
        )
        return

    if "height_m" not in df.columns:
        log(
            "[WARN] 'height_m' column missing; "
            "cannot make raw-height plot."
        )
        return
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df = df[np.isfinite(df["height_m"])]
    if df.empty:
        return

    ordered_keys = [
        "era5",
        "wtk",
        "hrrr",
        "wtk_led_conus",
        "wtk_led_climate",
        "gwa",
    ]
    label_map = {
        "era5": "ERA5",
        "wtk": "WTK",
        "hrrr": "HRRR",
        "wtk_led_conus": "WTK-LED\nCONUS",
        "wtk_led_climate": "WTK-LED\nClimate",
        "gwa": "GWA",
    }
    base_palette = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#B07AA1",
        "#76B7B2",
        "#59A14F",
    ]
    color_map = {
        k: base_palette[i] for i, k in enumerate(ordered_keys)
    }

    df["dataset"] = df["dataset"].astype(str)
    present = [
        k for k in ordered_keys if k in df["dataset"].unique()
    ]
    if not present:
        return

    finite_bias = df["bias"].to_numpy(dtype="float64")
    finite_bias = finite_bias[np.isfinite(finite_bias)]
    if finite_bias.size == 0:
        return
    L = float(np.nanmax(np.abs(finite_bias))) * 1.05
    y_min, y_max = -L, L

    h_vals = df["height_m"].to_numpy(dtype="float64")
    h_vals = h_vals[np.isfinite(h_vals)]
    if h_vals.size == 0:
        return
    x_min = 0.0
    x_max = max(120.0, np.ceil(float(np.nanmax(h_vals)) / 10.0) * 10.0)

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        sharey=True,
        figsize=(12.0, 5.0),
        dpi=300,
    )
    axes_flat = axes.flatten()

    for idx, d in enumerate(present):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        color = color_map.get(d, "#333333")
        label = label_map.get(d, d)

        sub = df[df["dataset"] == d]
        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub["height_m"].to_numpy(dtype="float64")
        y = sub["bias"].to_numpy(dtype="float64")

        ax.axhline(
            0.0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0
        )
        ax.grid(
            axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0
        )
        ax.scatter(
            x, y, color=color, s=10, alpha=0.9, edgecolors="none", zorder=1
        )
        ax.text(
            0.95,
            0.92,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)

    if len(present) < len(axes_flat):
        for ax in axes_flat[len(present):]:
            ax.set_visible(False)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if not ax.get_visible():
                continue
            if r < n_rows - 1:
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.tick_params(labelleft=False)

    for ax in axes[n_rows - 1, :]:
        if not ax.get_visible():
            continue
        ax.set_xlabel("Height (m)", fontsize=16)

    for r in range(n_rows):
        ax = axes[r, 0]
        if not ax.get_visible():
            continue
        ax.set_ylabel(r"Bias (m s$^{-1}$)", fontsize=16)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log(f"[INFO] Wrote height-bias raw-panel figure -> {out_path}")


def print_latex_rows(
    summary: pd.DataFrame, dataset_order: List[str]
) -> None:
    """Print LaTeX table rows for each dataset and subset."""

    def fmt(x: float) -> str:
        if x is None or not np.isfinite(x):
            return r"\dots"
        return f"{x:.2f}"

    print("% LaTeX rows for Table~\\ref{tab:dataset-summary}")
    for d in dataset_order:
        for subset in ["ASOS", "GS"]:
            row = summary[
                (summary["dataset"] == d)
                & (summary["subset"] == subset)
            ]
            if row.empty:
                mb = maa = medaa = mapa = np.nan
            else:
                r0 = row.iloc[0]
                mb = r0["median_bias"]
                maa = r0["mean_abs_bias"]
                medaa = r0["median_abs_bias"]
                mapa = r0["mean_abs_pct_bias"]

            label = DATASET_LATEX.get(d, d)
            print(
                f"{label:12s} & {subset:4s} & {fmt(mb)} & "
                f"{fmt(maa)} & {fmt(medaa)} & {fmt(mapa)} \\\\"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize dataset bias metrics (ASOS vs GS)."
    )
    ap.add_argument(
        "--infile",
        type=Path,
        default=Path(
            "combined_quantiles_long_with_topo_loocv_10km.csv"
        ),
        help="Long-format quantile table.",
    )
    ap.add_argument(
        "--gwa",
        type=Path,
        default=None,
        help="Optional CSV with per-site/height GWA means.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_out_table"),
        help="Directory for output.",
    )
    ap.add_argument(
        "--min_qrows",
        type=int,
        default=10,
        help="Minimum quantile rows per station_id+height_m.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="era5,wtk,wtk_led_climate,wtk_led_conus,hrrr,bchrrr",
        help="Comma-separated list of dataset column names.",
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

    log("[INFO] Aggregating to site/height means and biases ...")
    site = aggregate_site_means(
        df, dataset_cols=dataset_cols, min_qrows=args.min_qrows
    )

    dataset_keys = list(dataset_cols)
    if args.gwa is not None:
        site = merge_gwa_means(site, args.gwa)
        if "bias_gwa" in site.columns:
            dataset_keys.append("gwa")

    log("[INFO] Computing summary bias metrics for ASOS and GS ...")
    summary = compute_summary_metrics(
        site, dataset_keys=dataset_keys
    )

    out_csv = args.outdir / "dataset_summary.csv"
    summary.to_csv(out_csv, index=False)
    log(f"[INFO] Wrote dataset summary metrics -> {out_csv}")

    log(
        "[INFO] Computing height-stratified metrics for GS sites ..."
    )
    height_summary_gs = compute_height_metrics(
        site=site,
        dataset_keys=dataset_keys,
        subset_filter="GS",
        round_to=20,
    )

    out_height_csv = (
        args.outdir / "dataset_summary_by_height_gs.csv"
    )
    height_summary_gs.to_csv(out_height_csv, index=False)
    log(
        f"[INFO] Wrote GS-by-height summary metrics -> "
        f"{out_height_csv}"
    )

    plot_height_dependence(
        height_summary=height_summary_gs,
        out_path=args.outdir / "height_dependence_gs.png",
    )

    log(
        "[INFO] Preparing per-site bias distributions "
        "by height band for GS ..."
    )
    height_bias_gs = compute_height_bias_distributions(
        site=site,
        dataset_keys=dataset_keys,
        subset_filter="GS",
        round_to=20,
    )

    plot_height_bias_raw_panels(
        height_bias=height_bias_gs,
        out_path=args.outdir / "height_bias_boxplots_gs.png",
        subset_filter="GS",
    )

    default_order = [
        "era5",
        "wtk",
        "wtk_led_climate",
        "wtk_led_conus",
        "hrrr",
        "bchrrr",
        "gwa",
    ]
    present = set(summary["dataset"].unique())
    ordered = [d for d in default_order if d in present]

    print_latex_rows(summary, dataset_order=ordered)
    log("[INFO] Done.")


if __name__ == "__main__":
    main()
