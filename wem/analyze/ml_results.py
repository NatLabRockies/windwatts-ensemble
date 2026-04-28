#!/usr/bin/env python3
"""
GS-only analysis of bias and absolute bias vs observations for wind-resource datasets,
ML predictions, and (optionally) Global Wind Atlas (GWA).

Input  (default): combined_quantiles_long_with_topo_loocv_pred.csv   [LONG table with qnum 0..100]
Optional GWA input:  --gwa site_height_ws_avg_with_gwa.csv
                     (one row per station_id + height_m, with column 'gwa_interp' = mean WS at hub height)

Output (default dir): analysis_out_gs/
  - site_metrics_gs.csv
  - bias_*.png and absbias_*.png for ERA5/HRRR/WTK/WL_CONUS/WL_CLIM/ML/(GWA if provided)
  - ml_vs_<dataset>_delta_absbias.png  (delta|bias| maps, ML - dataset; includes GWA if provided)
  - bias_boxplots.png, abs_bias_boxplots.png
  - ml_parity_mean.png

Only Gold Standard (GS) sites are included in every step.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from mpl_toolkits.axes_grid1 import make_axes_locatable

# cartopy (optional for maps)
try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

from wem.utils.logging import log
from wem.utils.plotting import robust_limits, setup_cartopy_axes, symmetric_bias_limit
from wem.utils.sites import normalize_obs_type


# Re-export from consolidated location for backward compatibility
from wem.utils.quantiles import mean_from_quantile_long as mean_from_quantile_series

def pretty_boxplot(ax, data, labels, ylabel, title, zero_line=False):
    palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#9C755F", "#B07AA1"]
    colors = palette[:len(data)]
    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        vert=True,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, edgecolor="#333", linewidth=1.1, alpha=0.9)
    for elem in ["whiskers", "caps"]:
        for line in bp[elem]:
            line.set(color="#555", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="#222", linewidth=1.6)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#aaa", alpha=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    if zero_line:
        ax.axhline(0.0, color="#333", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=16)

# ───────────── cartopy maps ─────────────
def draw_map(df_pts: pd.DataFrame,
             title: str,
             cbar_label: str,
             out_path: Path,
             vmin: float | None = None,
             vmax: float | None = None,
             cmap: str = "viridis",
             diverging: bool = False,
             conus: bool = False,
             value_col: str = "bias",
             ne_res: str = "50m",
             marker: str = "o",
             ms: float = 18.0):

    if df_pts is None or df_pts.empty:
        log(f"[WARN] No data to plot for {title}")
        return
    if not HAS_CARTOPY:
        log("[WARN] Cartopy not available; skipping map.")
        return

    log(f"Rendering map: {title} -> {out_path.name}")
    fig, ax = setup_cartopy_axes(conus=conus, ne_res=ne_res)

    # Use function args if provided, otherwise fallback to defaults
    vmin = -2
    vmax = 2

    if diverging:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        sc = ax.scatter(df_pts["lon"], df_pts["lat"],
                        c=df_pts[value_col], s=ms, zorder=3,
                        cmap=cmap, norm=norm, linewidths=0.25, edgecolors="#333",
                        alpha=0.95, marker=marker, transform=ccrs.PlateCarree())
    else:
        sc = ax.scatter(df_pts["lon"], df_pts["lat"],
                        c=df_pts[value_col], s=ms, zorder=3,
                        cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.25, edgecolors="#333",
                        alpha=0.95, marker=marker, transform=ccrs.PlateCarree())

    title_text = '' # Renamed to avoid shadowing the argument
    ax.set_title(title_text, fontsize=13, pad=12)

    # --- COLORBAR FIX ---
    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)
    # Append axes to the right of size 3% with 0.1 pad
    # axes_class=plt.Axes is critical here to prevent the colorbar from inheriting
    # the Cartopy projection
    cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)

    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(cbar_label, fontsize=16)
    # --------------------

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ───────────── aggregation ─────────────
def aggregate_site_means(df: pd.DataFrame,
                         pred_col: str,
                         min_qrows: int = 10) -> pd.DataFrame:
    needed = ["station_id", "lat", "lon", "height_m", "qnum", "observation"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df = df.copy()
    df["qnum"] = pd.to_numeric(df["qnum"], errors="coerce")

    sources = ["observation", "era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate"]
    if pred_col in df.columns:
        sources.append(pred_col)

    rows = []
    for (sid, h), g in df.groupby(["station_id", "height_m"], sort=False):
        if np.isfinite(g["qnum"]).sum() < min_qrows:
            continue
        g = g.sort_values("qnum")
        row = {
            "station_id": str(sid),
            "height_m": float(g["height_m"].iloc[0]),
            "lat": float(g["lat"].iloc[0]),
            "lon": float(g["lon"].iloc[0]),
            "name": g["name"].iloc[0] if "name" in g.columns else str(sid),
            "observation_type": normalize_obs_type(str(g["observation_type"].iloc[0])) if "observation_type" in g.columns else "",
        }
        means: Dict[str, float] = {}
        for s in sources:
            if s in g.columns:
                means[s] = mean_from_quantile_series(
                    g["qnum"].to_numpy(),
                    pd.to_numeric(g[s], errors="coerce").to_numpy()
                )
        for k, v in means.items():
            row[f"mean_{k}"] = v
        m_obs = means.get("observation", np.nan)
        for k in ["era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate", pred_col]:
            if k in means:
                b = means[k] - m_obs if np.isfinite(means[k]) and np.isfinite(m_obs) else np.nan
                row[f"bias_{k}"] = b
                row[f"absbias_{k}"] = abs(b) if np.isfinite(b) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

# ───────────── plotting suites ─────────────
def plot_all_maps(site: pd.DataFrame, outdir: Path, conus: bool,
                  trim: float, ne_res: str, pred_col: str, include_gwa: bool):
    outdir.mkdir(parents=True, exist_ok=True)

    keys = ["era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate", pred_col]
    if include_gwa:
        keys.append("gwa")

    bias_cols = [f"bias_{k}" for k in keys if f"bias_{k}" in site.columns]
    abs_cols  = [c.replace("bias_", "absbias_") for c in bias_cols]

    bias_series = [site[c] for c in bias_cols]
    abs_series  = [site[c] for c in abs_cols]

    L = symmetric_bias_limit(bias_series, trim=trim)
    vmin_b, vmax_b = -L, +L
    _, hi_abs = robust_limits(abs_series, trim=trim)
    vmin_a, vmax_a = 0.0, hi_abs

    def sub(df: pd.DataFrame, valcol: str) -> pd.DataFrame:
        x = df[["lat", "lon", valcol]].copy()
        x[valcol] = pd.to_numeric(x[valcol], errors="coerce")
        x = x[np.isfinite(x[valcol])]
        return x

    labels = {
        "era5": "ERA5",
        "hrrr": "HRRR",
        "wtk": "WTK",
        "wtk_led_conus": "WTK-LED CONUS",
        "wtk_led_climate": "WTK-LED Climate",
        pred_col: "WEM",
        "gwa": "GWA",
    }

    for key in keys:
        bcol = f"bias_{key}"
        if bcol in site.columns:
            draw_map(
                sub(site, bcol),
                title=f"Bias ({labels[key]} - Obs), GS sites",
                cbar_label=r"Bias ($m\,s^{-1}$)",
                out_path=outdir / f"bias_{key}.png",
                vmin=vmin_b, vmax=vmax_b,
                cmap="RdBu", diverging=True,
                conus=conus, value_col=bcol, ne_res=ne_res,
                marker="o", ms=18.0
            )
    for key in keys:
        acol = f"absbias_{key}"
        if acol in site.columns:
            draw_map(
                sub(site, acol),
                title=f"Absolute Bias ({labels[key]} vs Obs), GS sites",
                cbar_label=r"$|$Bias$|$ ($m\,s^{-1}$)",
                out_path=outdir / f"absbias_{key}.png",
                vmin=vmin_a, vmax=vmax_a,
                cmap="magma", diverging=False,
                conus=conus, value_col=acol, ne_res=ne_res,
                marker="o", ms=18.0
            )

def plot_boxplots(site: pd.DataFrame, outdir: Path, pred_col: str, include_gwa: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    order = ["era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate", pred_col]
    if include_gwa:
        order.append("gwa")
    labels = {
        "era5": "ERA5",
        "hrrr": "HRRR",
        "wtk": "WTK",
        "wtk_led_conus": "WTK-LED CONUS",
        "wtk_led_climate": "WTK-LED Climate",
        pred_col: "WEM",
        "gwa": "GWA",
    }

    # ---- Bias (sort by decreasing median |bias|) ----
    entries_b = []
    for k in order:
        col = f"bias_{k}"
        if col in site.columns:
            x = pd.to_numeric(site[col], errors="coerce").to_numpy(dtype="float64")
            x = x[np.isfinite(x)]
            if x.size > 0:
                med_abs = float(np.nanmedian(np.abs(x)))  # distance from zero
                entries_b.append((labels[k], x, med_abs))

    if entries_b:
        entries_b.sort(key=lambda t: t[2], reverse=True)
        lab_b  = [e[0] for e in entries_b]
        data_b = [e[1] for e in entries_b]

        fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
        pretty_boxplot(
            ax1, data_b, lab_b,
            ylabel=r"Bias ($m\,s^{-1}$)",
            title='',
            zero_line=True
        )
        fig1.tight_layout()
        fig1.savefig(outdir / "bias_boxplots.png", bbox_inches="tight", dpi=300)
        plt.close(fig1)

    # ---- Absolute bias (sort by decreasing median |bias|) ----
    entries_a = []
    for k in order:
        col = f"absbias_{k}"
        if col in site.columns:
            x = pd.to_numeric(site[col], errors="coerce").to_numpy(dtype="float64")
            x = x[np.isfinite(x)]
            if x.size > 0:
                med = float(np.nanmedian(x))
                entries_a.append((labels[k], x, med))

    if entries_a:
        entries_a.sort(key=lambda t: t[2], reverse=True)
        lab_a  = [e[0] for e in entries_a]
        data_a = [e[1] for e in entries_a]

        fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
        pretty_boxplot(
            ax2, data_a, lab_a,
            ylabel=r"$|$Bias$|$ ($m\,s^{-1}$)",
            title='',
            zero_line=False
        )
        fig2.tight_layout()
        fig2.savefig(outdir / "abs_bias_boxplots.png", bbox_inches="tight", dpi=300)
        plt.close(fig2)


def plot_ml_parity(site: pd.DataFrame, outdir: Path, pred_col: str):
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract observed and predicted means
    y_true = pd.to_numeric(site.get("mean_observation", np.nan), errors="coerce").to_numpy(dtype="float64")
    y_pred = pd.to_numeric(site.get(f"mean_{pred_col}", np.nan), errors="coerce").to_numpy(dtype="float64")

    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(good):
        log("[WARN] No ML parity data to plot.")
        return

    t = y_true[good]
    p = y_pred[good]
    bias = p - t

    # Axis limits based on data (similar spirit to the original parity plot)
    all_vals = np.concatenate([t, p])
    lo = float(np.nanpercentile(all_vals, 1))
    hi = float(np.nanpercentile(all_vals, 99))
    pad = 0.05 * (hi - lo)
    xylim_min = 0
    xylim_max = 10

    # Symmetric color scale for bias
    # Use a high percentile to avoid being dominated by outliers
    L_bias = float(np.nanpercentile(np.abs(bias), 99))
    if not np.isfinite(L_bias) or L_bias <= 0:
        L_bias = float(np.nanmax(np.abs(bias))) if np.any(np.isfinite(bias)) else 1.0

    df_xy = pd.DataFrame({
        "x_obs": t,
        "y_model": p,
        "bias": bias,
    })

    out_path = outdir / "ml_parity_mean.png"
    log(f"Rendering scatter (Obs vs Dataset): ML parity -> {out_path.name}")

    # --- Style copied from draw_ws_scatter ---
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=300)

    norm = TwoSlopeNorm(vmin=-2, vcenter=0.0, vmax=2)
    sc = ax.scatter(
        df_xy["x_obs"],
        df_xy["y_model"],
        c=df_xy["bias"],
        cmap="RdBu_r",
        norm=norm,
        s=16,
        linewidths=0.2,
        edgecolors="#333333",
        alpha=0.8,
    )

    # 1:1 line and framing
    ax.plot(
        [xylim_min, xylim_max],
        [xylim_min, xylim_max],
        linestyle="--",
        linewidth=1.0,
        color="#444444",
        alpha=0.9,
    )
    ax.set_xlim(xylim_min, xylim_max)
    ax.set_ylim(xylim_min, xylim_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"Observed mean wind speed ($m\,s^{-1}$)", fontsize=16)
    ax.set_ylabel(r"Dataset mean wind speed ($m\,s^{-1}$)", fontsize=16)
    ax.grid(True, linestyle="--", linewidth=0.6, color="#aaaaaa", alpha=0.6)

    cbar = plt.colorbar(sc, ax=ax, pad=0.012, fraction=0.046)
    cbar.set_label(r"Bias ($m\,s^{-1}$)", fontsize=16)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)




def plot_ml_vs_dataset_diff_maps(site: pd.DataFrame,
                                outdir: Path,
                                pred_col: str,
                                conus: bool,
                                trim: float,
                                ne_res: str,
                                include_gwa: bool) -> None:
    """
    For each dataset D, map delta|bias| = |bias_ML| - |bias_D|.
      > 0  (red): ML worse (higher absolute bias)
      = 0  (white): equal
      < 0  (blue): ML better (lower absolute bias)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for k in ["era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate", "gwa" if include_gwa else None]:
        if k and f"absbias_{k}" in site.columns:
            datasets.append(k)
    if not datasets or f"absbias_{pred_col}" not in site.columns:
        log("[WARN] Missing absbias columns for ML vs dataset deltas; skipping.")
        return

    diff_cols = []
    for k in datasets:
        dcol = f"delta_absbias_ml_vs_{k}"
        site[dcol] = pd.to_numeric(site[f"absbias_{pred_col}"], errors="coerce") \
                   - pd.to_numeric(site[f"absbias_{k}"], errors="coerce")
        diff_cols.append(dcol)

    diffs_series = [site[c] for c in diff_cols]
    L = symmetric_bias_limit(diffs_series, trim=trim)
    vmin, vmax = -L, +L

    name_map = {
        "era5": "ERA5",
        "hrrr": "HRRR",
        "wtk": "WTK",
        "wtk_led_conus": "WTK-LED CONUS",
        "wtk_led_climate": "WTK-LED Climate",
        "gwa": "GWA",
    }

    def sub(df: pd.DataFrame, valcol: str) -> pd.DataFrame:
        x = df[["lat", "lon", valcol]].copy()
        x[valcol] = pd.to_numeric(x[valcol], errors="coerce")
        return x[np.isfinite(x[valcol])]

    for k in datasets:
        dcol = f"delta_absbias_ml_vs_{k}"
        draw_map(
            sub(site, dcol),
            title=f"delta|bias| (ML - {name_map[k]}) at GS sites",
            cbar_label=r"$\Delta|$bias$|$ ($m\,s^{-1}$)  [>0 ML worse, <0 ML better]",
            out_path=outdir / f"ml_vs_{k}_delta_absbias.png",
            vmin=vmin, vmax=vmax,
            cmap="RdBu_r",
            diverging=True,
            conus=conus,
            value_col=dcol,
            ne_res=ne_res,
            marker="o",
            ms=18.0,
        )

def plot_ml_bias_map(site: pd.DataFrame,
                     outdir: Path,
                     pred_col: str,
                     conus: bool,
                     trim: float,
                     ne_res: str) -> None:
    """
    Single GS map of ML bias at GS sites, using big diamond markers.

    Bias is (mean_{pred_col} - mean_observation), color scale is symmetric
    and pooled across all dataset biases (ERA5/HRRR/WTK/WTK-LED/ML/[GWA if present])
    so it matches the GS bias maps.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    bias_col = f"bias_{pred_col}"

    # Ensure we have the ML bias column
    if bias_col not in site.columns:
        if "mean_observation" not in site.columns or f"mean_{pred_col}" not in site.columns:
            log(f"[WARN] Cannot compute ML bias: missing mean_observation or mean_{pred_col}.")
            return
        log(f"[INFO] Computing {bias_col} from mean_{pred_col} and mean_observation.")
        site[bias_col] = (
            pd.to_numeric(site[f"mean_{pred_col}"], errors="coerce")
            - pd.to_numeric(site["mean_observation"], errors="coerce")
        )

    # Use the same pooled symmetric bias limits as the GS maps
    keys = ["era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate", pred_col]
    if "bias_gwa" in site.columns:
        keys.append("gwa")

    bias_series = [site[f"bias_{k}"] for k in keys if f"bias_{k}" in site.columns]
    if not bias_series:
        log("[WARN] No bias series found to set color limits; using default +/-2 m/s.")
        L = 2.0
    else:
        L = symmetric_bias_limit(bias_series, trim=trim)
    vmin_b, vmax_b = -L, +L
    log(f"[ML bias map] Symmetric limits from pooled biases: [{vmin_b:.2f}, {vmax_b:.2f}]")

    # Subset to finite ML bias
    df_map = site[["lat", "lon", bias_col]].copy()
    df_map[bias_col] = pd.to_numeric(df_map[bias_col], errors="coerce")
    df_map = df_map[np.isfinite(df_map[bias_col])]
    if df_map.empty:
        log("[WARN] No finite ML bias values to plot.")
        return

    # Draw with same basemap style, but big diamonds for GS sites
    draw_map(
        df_pts=df_map,
        title="ML Bias (ML - Obs), GS sites",
        cbar_label=r"Bias ($m\,s^{-1}$)",
        out_path=outdir / "ml_bias_map.png",
        vmin=vmin_b,
        vmax=vmax_b,
        cmap="RdBu_r",
        diverging=True,
        conus=conus,
        value_col=bias_col,
        ne_res=ne_res,
        marker="D",   # big diamonds
        ms=42.0,      # size similar to your other GS diamond maps
    )

def plot_bias_vs_quantile(df: pd.DataFrame,
                          outdir: Path,
                          pred_col: str) -> None:
    """
    Faceted bias-vs-quantile plots using IQR (25th-75th percentile).

    For each dataset and each quantile q, compute the distribution of
    bias across GS rows:
        bias(q) = dataset(q) - observation(q)

    Then, for each dataset, plot the median, 25th, and 75th percentile
    vs q in a 2x3 grid of subplots, shading between the 25th and 75th
    percentiles. Dataset label is shown in bold in the bottom-left
    corner of each panel.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if "qnum" not in df.columns or "observation" not in df.columns:
        log("[WARN] Missing 'qnum' or 'observation' in long table; "
            "cannot plot bias vs quantile.")
        return

    data = df.copy()
    data["qnum"] = pd.to_numeric(data["qnum"], errors="coerce")
    data = data[np.isfinite(data["qnum"])]

    if data.empty:
        log("[WARN] No valid qnum rows for bias-vs-quantile plot.")
        return

    obs = pd.to_numeric(data["observation"], errors="coerce")

    # Candidate datasets that have quantiles in the long table
    candidates = ["era5", "wtk", "hrrr", "wtk_led_conus", "wtk_led_climate", pred_col]

    label_map = {
        "era5": "ERA5",
        "wtk": "WTK",
        "hrrr": "HRRR",
        "wtk_led_conus": "WTK-LED CONUS",
        "wtk_led_climate": "WTK-LED Climate",
        pred_col: "WEM",
    }

    # Fixed palette in desired order
    base_palette = ["#4E79A7", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2", "#59A14F"]
    color_map = {
        "era5": base_palette[0],
        "wtk": base_palette[1],
        "hrrr": base_palette[2],
        "wtk_led_conus": base_palette[3],
        "wtk_led_climate": base_palette[4],
        pred_col: base_palette[5],
    }

    # Compute (q25, median, q75) bias(q) for each dataset
    stats_by_dataset: Dict[str, pd.DataFrame] = {}
    for col in candidates:
        if col not in data.columns:
            continue

        vals = pd.to_numeric(data[col], errors="coerce")
        bias = vals - obs

        tmp = pd.DataFrame({"qnum": data["qnum"], "bias": bias})
        g = tmp.groupby("qnum", as_index=True)["bias"]

        qstats = (
            g.quantile([0.25, 0.5, 0.75])
             .unstack()
             .rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
        )
        qstats = (
            qstats.replace([np.inf, -np.inf], np.nan)
                  .dropna(how="all")
                  .sort_index()
        )

        if not qstats.empty:
            stats_by_dataset[col] = qstats

    if not stats_by_dataset:
        log("[WARN] No datasets with quantiles found for bias-vs-quantile plot.")
        return

    # Enforce panel order and keep only datasets we actually have
    ordered_keys = ["era5", "wtk", "hrrr", "wtk_led_conus", "wtk_led_climate", pred_col]
    keys = [k for k in ordered_keys if k in stats_by_dataset]
    n_panels = len(keys)
    if n_panels == 0:
        log("[WARN] No datasets with valid bias stats for bias-vs-quantile plot.")
        return

    # Global y-limits across all datasets (from q25/median/q75)
    all_vals_list = []
    for k in keys:
        arr = stats_by_dataset[k][["q25", "median", "q75"]].to_numpy().ravel()
        all_vals_list.append(arr)
    all_vals = np.concatenate(all_vals_list)
    all_vals = all_vals[np.isfinite(all_vals)]

    if all_vals.size == 0:
        log("[WARN] Non-finite y-limits for bias-vs-quantile plot.")
        return

    y_lo = float(all_vals.min())
    y_hi = float(all_vals.max())
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    y_min = y_lo - margin
    y_max = y_hi + margin

    # 2 rows x 3 columns of subplots
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), dpi=150,
                             sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        stats = stats_by_dataset[key]

        q = stats.index.to_numpy()
        q25 = stats["q25"].to_numpy()
        med = stats["median"].to_numpy()
        q75 = stats["q75"].to_numpy()

        color = color_map.get(key, "#cccccc")

        # Shade between 25th and 75th percentile
        ax.fill_between(
            q,
            q25,
            q75,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )

        # Median line
        ax.plot(
            q,
            med,
            color=color,
            linewidth=2.0 if key == pred_col else 1.6,
        )

        # Optional: dashed lines for q25 and q75
        ax.plot(
            q,
            q25,
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(
            q,
            q75,
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
        )

        # Zero line
        ax.axhline(0.0, color="#444444", linestyle="--",
                   linewidth=0.8, alpha=0.8)

        ax.set_xlim(0, 100)
        ax.set_ylim(-3, 3)
        ax.grid(True, linestyle="--", linewidth=0.5,
                color="#aaaaaa", alpha=0.6)

        # Bold label in bottom-left corner of panel
        ax.text(
            0.03, 0.04,
            label_map.get(key, key),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        # Clean up spines
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Hide any unused panels (if fewer than 6 datasets are present)
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    # Axis labels: y on left column, x on bottom row
    for ax in (axes[0], axes[3]):  # left column
        if ax.get_visible():
            ax.set_ylabel(r"Bias ($m\,s^{-1}$)", fontsize=14)
    for ax in (axes[3], axes[4], axes[5]):  # bottom row
        if ax.get_visible():
            ax.set_xlabel("Quantile q (%)", fontsize=14)
            ax.set_xticks(np.arange(0, 101, 20))

    fig.tight_layout()
    out_path = outdir / "bias_vs_quantile_faceted_iqr.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log(f"[INFO] Wrote faceted IQR bias-vs-quantile plot -> {out_path}")



def print_mean_abs_bias_table(site: pd.DataFrame,
                              pred_col: str,
                              include_gwa: bool,
                              outdir: Optional[Path] = None) -> None:
    """
    Print and optionally save a small table of mean absolute bias
    (averaged over GS site-height combinations) for each dataset,
    including GWA if present.

    Uses the site-level columns:
      absbias_era5, absbias_wtk, absbias_wtk_led_climate,
      absbias_wtk_led_conus, absbias_hrrr, absbias_<pred_col>,
      and absbias_gwa (if include_gwa=True).
    """
    # Order similar to the paper table, with ML (pred_col) last
    order = ["era5", "wtk", "wtk_led_climate", "wtk_led_conus", "hrrr", pred_col]
    if include_gwa:
        order.append("gwa")

    labels = {
        "era5": "ERA5",
        "wtk": "WTK",
        "wtk_led_climate": "WTK-LED Climate",
        "wtk_led_conus": "WTK-LED CONUS",
        "hrrr": "HRRR",
        pred_col: "WEM",
        "gwa": "GWA",
    }

    rows = []
    for key in order:
        abs_col = f"absbias_{key}"
        if abs_col not in site.columns:
            continue

        vals = pd.to_numeric(site[abs_col], errors="coerce").to_numpy(dtype="float64")
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        mean_abs = float(np.nanmean(vals))
        median_abs = float(np.nanmedian(vals))
        n = int(vals.size)

        rows.append({
            "dataset": labels.get(key, key),
            "mean_abs_bias_ms": mean_abs,
            "median_abs_bias_ms": median_abs,
            "n_site_heights": n,
        })

    if not rows:
        log("[WARN] No absbias_* columns found; cannot print summary table.")
        return

    summary = pd.DataFrame(rows)

    log("[INFO] Mean absolute bias over GS site-height combinations (m/s):")
    with pd.option_context("display.max_rows", None,
                           "display.float_format", lambda x: f"{x:6.3f}"):
        print(summary.to_string(index=False))

    # Also write to CSV for convenience
    if outdir is not None:
        out_path = outdir / "mean_abs_bias_summary_gs.csv"
        summary.to_csv(out_path, index=False)
        log(f"[INFO] Wrote mean-absolute-bias summary -> {out_path}")




# ───────────── main ─────────────
def main():
    ap = argparse.ArgumentParser(description="GS-only maps & boxplots of bias/abs-bias for datasets, GWA (optional), and ML vs observations.")
    ap.add_argument("--infile",  type=Path, default=Path("ml_results_optimized_hyperparams.csv"))
    ap.add_argument("--outdir",  type=Path, default=Path("analysis_out_opt_hyperparams"))
    ap.add_argument("--pred_col", type=str, default="pred_observation")
    ap.add_argument("--min_qrows", type=int, default=10)
    ap.add_argument("--conus", action="store_true")
    ap.add_argument("--trim", type=float, default=0.02)
    ap.add_argument("--ne_res", type=str, default="50m", choices=["110m","50m","10m"])
    ap.add_argument("--gwa", type=Path, default=None,
                    help="Optional CSV with per-site/height Global Wind Atlas means (expects columns: station_id,height_m,gwa_interp).")
    args = ap.parse_args()

    if not args.infile.exists():
        raise FileNotFoundError(args.infile)
    args.outdir.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Loading long table: {args.infile}")
    df = pd.read_csv(args.infile, dtype={"station_id": str}, low_memory=False)

    # ---- GS filter BEFORE aggregation (apples-to-apples) ----
    if "observation_type" not in df.columns:
        raise ValueError("Input must contain 'observation_type' column to filter GS sites.")
    typ = df["observation_type"].astype(str).map(normalize_obs_type)
    gs_ids = df.loc[typ.eq("GS"), "station_id"].astype(str).unique()
    log(f"[INFO] GS stations found: {len(gs_ids)}")
    df = df[df["station_id"].astype(str).isin(gs_ids)].copy()
    log(f"[INFO] Rows after GS-only filter: {len(df)}")

    # ---- Aggregate to site-level means/biases (LONG -> site_height) ----
    log("[INFO] Aggregating to site-level means & biases (GS only) ...")
    site = aggregate_site_means(df, pred_col=args.pred_col, min_qrows=args.min_qrows)

    # ---- Optional: merge GWA (site/height means) ----
    include_gwa = False
    if args.gwa is not None:
        if not args.gwa.exists():
            raise FileNotFoundError(f"GWA CSV not found: {args.gwa}")
        log(f"[INFO] Merging GWA site/height means from: {args.gwa.name}")
        gwa = pd.read_csv(args.gwa, dtype={"station_id": str}, low_memory=False)

        if "station_id" not in gwa.columns or "height_m" not in gwa.columns:
            raise ValueError("GWA CSV must include 'station_id' and 'height_m' columns.")

        gwa_col = "gwa_interp" if "gwa_interp" in gwa.columns else None
        if gwa_col is None:
            for c in ["gwa", "mean_gwa"]:
                if c in gwa.columns:
                    gwa_col = c
                    break
        if gwa_col is None:
            raise ValueError("GWA CSV must contain 'gwa_interp' (mean WS at hub height).")

        # Keep ONLY the columns needed for the merge to avoid creating lat_x/lon_x
        gwa = gwa[["station_id", "height_m", gwa_col]].copy()
        gwa["height_m"] = pd.to_numeric(gwa["height_m"], errors="coerce")
        gwa = gwa.dropna(subset=["station_id", "height_m"]).drop_duplicates(subset=["station_id","height_m"])

        # Merge without bringing in lat/lon, so site['lat'], site['lon'] remain unchanged
        site = site.merge(gwa.rename(columns={gwa_col: "mean_gwa"}),
                        on=["station_id", "height_m"], how="left")

        if "mean_gwa" in site.columns and "mean_observation" in site.columns:
            b = site["mean_gwa"] - site["mean_observation"]
            site["bias_gwa"] = b
            site["absbias_gwa"] = np.abs(b)
            # Keep only site-height rows that have GWA coverage
            before = len(site)
            site = site.loc[site["mean_gwa"].notna()].copy()
            log(f"[INFO] Filtered to GWA-covered site/heights: {before} -> {len(site)}")

            include_gwa = True
            log(f"[INFO] GWA merged: non-null rows = {int(site['mean_gwa'].notna().sum()):,}")
        else:
            log("[WARN] Could not compute GWA bias (missing mean_gwa or mean_observation).")


    # Save site-level metrics
    site_out = args.outdir / "site_metrics_gs.csv"
    site.to_csv(site_out, index=False)
    log(f"[INFO] Wrote GS site metrics -> {site_out}")

    # Maps
    plot_all_maps(site, outdir=args.outdir, conus=args.conus, trim=args.trim, ne_res=args.ne_res,
                  pred_col=args.pred_col, include_gwa=include_gwa)

    # Single ML bias map with big GS diamonds
    plot_ml_bias_map(site, outdir=args.outdir, pred_col=args.pred_col,
                     conus=args.conus, trim=args.trim, ne_res=args.ne_res)

    # ML vs dataset delta|bias| maps (red = ML worse, blue = ML better) -- includes GWA if present
    plot_ml_vs_dataset_diff_maps(
        site=site,
        outdir=args.outdir,
        pred_col=args.pred_col,
        conus=args.conus,
        trim=args.trim,
        ne_res=args.ne_res,
        include_gwa=include_gwa,
    )

    # Bias vs quantile curves (uses long GS-only table)
    plot_bias_vs_quantile(df, outdir=args.outdir, pred_col=args.pred_col)

    # Boxplots
    plot_boxplots(site, outdir=args.outdir, pred_col=args.pred_col, include_gwa=include_gwa)

    # ML parity (mean)
    if f"mean_{args.pred_col}" in site.columns and "mean_observation" in site.columns:
        plot_ml_parity(site, outdir=args.outdir, pred_col=args.pred_col)
    else:
        log("[WARN] Missing mean columns for ML parity; skipping.")

    # Save site-level metrics
    site_out = args.outdir / "site_metrics_gs.csv"
    site.to_csv(site_out, index=False)
    log(f"[INFO] Wrote GS site metrics -> {site_out}")

    # Print table of mean absolute bias (GS site-height level)
    print_mean_abs_bias_table(
        site=site,
        pred_col=args.pred_col,
        include_gwa=include_gwa,
        outdir=args.outdir,
    )


    log(f"[INFO] Done. Outputs in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
