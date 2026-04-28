"""Pre-ML quantile-level visualization of wind resource datasets.

Produces US maps, obs-vs-model scatters, and bias boxplots from
per-station quantile CSVs — for both ASOS and Gold Standard cohorts,
with optional GWA (Global Wind Atlas) overlay.

Three output groups:
  A) ASOS set — outlier-filtered using ERA5−Obs bias
  B) Gold Standard set — no outlier removal
  C) Overlay set — ASOS circles + GS diamonds on shared maps

Merged from dev/create_maps.py + dev/analyze_datasets_gwa.py.

CLI entry point: ``wem-quantile-maps``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Cartopy (optional for maps)
try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

from wem.utils.columns import choose_col, find_qcols
from wem.utils.logging import log
from wem.utils.plotting import robust_limits, setup_cartopy_axes, symmetric_bias_limit
from wem.utils.quantiles import mean_from_quantiles_row as mean_from_quantiles

if HAS_CARTOPY:
    import cartopy.feature as cfeature
    from cartopy.feature import NaturalEarthFeature


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_quantile_file(path: Path, label: str) -> pd.DataFrame:
    """Load a quantile CSV and compute per-station mean wind speed.

    Returns a DataFrame with columns:
    ``station_id``, ``name``, ``lat``, ``lon``, ``mean_ws``
    (plus ``height_m`` when present in the source file).
    """
    if not path.exists():
        raise FileNotFoundError(path)
    log(f"Loading {label} file: {path}")
    df = pd.read_csv(path)
    id_col = choose_col(df, ["station_id", "STATION", "site_id", "id"])
    lat_col = choose_col(df, ["lat", "LAT", "Latitude"])
    lon_col = choose_col(df, ["lon", "LON", "Longitude"])
    name_col = choose_col(
        df, ["name", "NAME", "station_name", "STATION NAME", "site_name"]
    )
    if not (id_col and lat_col and lon_col):
        raise ValueError(
            f"{label}: required columns not found (need station_id/lat/lon)."
        )
    qcols = find_qcols(df)
    if len(qcols) < 50:
        raise ValueError(
            f"{label}: expected quantile columns q000..q100; found {len(qcols)}."
        )
    df = df.rename(
        columns={
            id_col: "station_id",
            lat_col: "lat",
            lon_col: "lon",
            (name_col or id_col): "name",
        }
    )
    log(f"Computing mean wind speed from {len(qcols)} quantiles for {label}...")
    df["mean_ws"] = df.apply(lambda r: mean_from_quantiles(r, qcols), axis=1)
    df = df.dropna(subset=["lat", "lon", "mean_ws"])
    cols = ["station_id", "name", "lat", "lon", "mean_ws"]
    if "height_m" in df.columns:
        cols.insert(1, "height_m")
    return df[cols].copy()


def load_gwa_file(path: Optional[Path], label: str) -> pd.DataFrame:
    """Load a GWA-per-site CSV with ``gwa_interp`` as mean wind speed.

    Returns an empty DataFrame if *path* is None or the file is missing.
    """
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        log(f"[WARN] {label} GWA file not found: {path} -- skipping.")
        return pd.DataFrame()
    log(f"Loading {label} GWA file: {path}")
    df = pd.read_csv(path)
    need = ["station_id", "lat", "lon", "height_m", "gwa_interp"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{label} GWA file missing columns: {miss}")
    name_col = choose_col(df, ["name", "NAME", "station_name"])
    if name_col and name_col != "name":
        df = df.rename(columns={name_col: "name"})
    if "name" not in df.columns:
        df["name"] = df["station_id"]
    for c in ["lat", "lon", "height_m", "gwa_interp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height_m", "gwa_interp"]).copy()
    df["mean_ws"] = df["gwa_interp"].astype(float)
    return df[["station_id", "height_m", "name", "lat", "lon", "mean_ws"]].copy()


# ---------------------------------------------------------------------------
# Bias computation
# ---------------------------------------------------------------------------


def build_bias_df(
    df_model: pd.DataFrame,
    df_obs: pd.DataFrame,
    label: str,
    join_cols: List[str],
) -> pd.DataFrame:
    """Merge model and obs on *join_cols* and compute bias = model - obs.

    Returns a DataFrame with ``station_id``, ``name``, ``lat``, ``lon``,
    ``bias``.  Returns an empty DataFrame if no overlap.
    """
    b = pd.merge(df_model, df_obs, on=join_cols, suffixes=("_model", "_obs"))
    if b.empty:
        log(f"{label}: no overlap with Obs; skipping bias map.")
        return b
    b["bias"] = b["mean_ws_model"] - b["mean_ws_obs"]
    return pd.DataFrame(
        {
            "station_id": b["station_id"],
            "name": b.get("name_model", b.get("name_obs", "")),
            "lat": b.get("lat_model", b["lat_obs"]),
            "lon": b.get("lon_model", b["lon_obs"]),
            "bias": b["bias"].astype(float),
        }
    )


def filter_by_bias(
    df_obs: pd.DataFrame,
    df_era5: pd.DataFrame,
    bias_trim: float,
) -> Set[str]:
    """Return station_ids whose ERA5-Obs bias falls within trimmed limits."""
    bias_e = pd.merge(
        df_era5, df_obs, on="station_id", suffixes=("_era5", "_obs")
    )
    if bias_e.empty:
        return set()
    bias_e["bias"] = bias_e["mean_ws_era5"] - bias_e["mean_ws_obs"]
    L = symmetric_bias_limit([bias_e["bias"]], trim=bias_trim)
    keep = bias_e.loc[
        np.isfinite(bias_e["bias"]) & bias_e["bias"].between(-L, L),
        "station_id",
    ]
    return set(keep)


# ---------------------------------------------------------------------------
# XY scatter helpers
# ---------------------------------------------------------------------------


def _build_xy_bias(
    df_model: pd.DataFrame,
    df_obs: pd.DataFrame,
    join_cols: List[str],
) -> pd.DataFrame:
    """Merge and return x=obs, y=model, bias=y-x."""
    m = pd.merge(df_model, df_obs, on=join_cols, suffixes=("_model", "_obs"))
    if m.empty:
        return m
    out = pd.DataFrame(
        {
            "x_obs": pd.to_numeric(m["mean_ws_obs"], errors="coerce"),
            "y_model": pd.to_numeric(m["mean_ws_model"], errors="coerce"),
        }
    )
    out["bias"] = out["y_model"] - out["x_obs"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["x_obs", "y_model", "bias"]
    )
    return out


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------


def draw_map(
    df_pts: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    diverging: bool = False,
    conus: bool = False,
    value_col: str = "mean_ws",
    ne_res: str = "50m",
    marker: str = "o",
    ms: float = 18.0,
    edgecolors: str = "#444",
    linewidths: float = 0.2,
    alpha: float = 0.9,
) -> None:
    """Scatter-map on a cartopy CONUS basemap."""
    if df_pts is None or df_pts.empty:
        log(f"{title}: no data; skipping.")
        return
    if not HAS_CARTOPY:
        log("[WARN] Cartopy not available; skipping map.")
        return
    log(f"Rendering map: {title} -> {out_path.name}")
    fig, ax = setup_cartopy_axes(conus=conus, ne_res=ne_res)

    kw: Dict = dict(
        s=ms,
        linewidths=linewidths,
        edgecolors=edgecolors,
        alpha=alpha,
        marker=marker,
        zorder=3,
        transform=ccrs.PlateCarree(),
    )

    if diverging:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        sc = ax.scatter(
            df_pts["lon"],
            df_pts["lat"],
            c=df_pts[value_col],
            cmap=cmap,
            norm=norm,
            **kw,
        )
    else:
        sc = ax.scatter(
            df_pts["lon"],
            df_pts["lat"],
            c=df_pts[value_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kw,
        )

    fig.tight_layout(rect=[0.0, 0.0, 0.93, 1.0])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(cbar_label)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_overlay_map(
    df_base: pd.DataFrame,
    df_overlay: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    diverging: bool,
    conus: bool,
    value_col: str,
    ne_res: str,
) -> None:
    """Dual-marker map: ASOS (circles) + Gold Std (diamonds)."""
    if not HAS_CARTOPY:
        log("[WARN] Cartopy not available; skipping overlay map.")
        return
    log(f"Rendering overlay: {title} -> {out_path.name}")
    fig, ax = setup_cartopy_axes(conus=conus, ne_res=ne_res)

    if diverging:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        sc = ax.scatter(
            df_base["lon"],
            df_base["lat"],
            c=df_base[value_col],
            cmap=cmap,
            norm=norm,
            s=22,
            edgecolors="#333",
            linewidths=0.25,
            alpha=0.9,
            marker="o",
            zorder=3,
            transform=ccrs.PlateCarree(),
        )
        ax.scatter(
            df_overlay["lon"],
            df_overlay["lat"],
            c=df_overlay[value_col],
            cmap=cmap,
            norm=norm,
            s=42,
            edgecolors="#111",
            linewidths=0.35,
            alpha=0.95,
            marker="D",
            zorder=4,
            transform=ccrs.PlateCarree(),
        )
    else:
        sc = ax.scatter(
            df_base["lon"],
            df_base["lat"],
            c=df_base[value_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=22,
            edgecolors="#333",
            linewidths=0.25,
            alpha=0.9,
            marker="o",
            zorder=3,
            transform=ccrs.PlateCarree(),
        )
        ax.scatter(
            df_overlay["lon"],
            df_overlay["lat"],
            c=df_overlay[value_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=42,
            edgecolors="#111",
            linewidths=0.35,
            alpha=0.95,
            marker="D",
            zorder=4,
            transform=ccrs.PlateCarree(),
        )

    cbar = plt.colorbar(sc, ax=ax, pad=0.015, fraction=0.035)
    cbar.set_label(cbar_label)

    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="ASOS set",
            markerfacecolor="#888888",
            markeredgecolor="#333333",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            label="Gold Std",
            markerfacecolor="#888888",
            markeredgecolor="#111111",
            markersize=7,
        ),
    ]
    ax.legend(
        handles=legend_elems, loc="lower left", fontsize=9, frameon=True, framealpha=0.9
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_bias_boxplots(
    bias_frames: Dict[str, pd.DataFrame],
    out_signed: Path,
    out_abs: Path,
) -> None:
    """Signed + absolute bias boxplots, sorted by median."""
    palette = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#B07AA1",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#AF7AA1",
    ]
    order = ["ERA5", "WTK", "HRRR", "WTK-LED CONUS", "WTK-LED Climate", "GWA"]
    labels, data = [], []
    for lbl in order:
        dfb = bias_frames.get(lbl)
        if dfb is not None and not dfb.empty and "bias" in dfb.columns:
            x = pd.to_numeric(dfb["bias"], errors="coerce").to_numpy(dtype="float64")
            x = x[np.isfinite(x)]
            if x.size > 0:
                labels.append(lbl)
                data.append(x)
    if not data:
        log("No bias data available for box plots; skipping.")
        return

    colors = palette[: len(data)]
    color_map = dict(zip(labels, colors))

    def _boxplot(ax, d, l, c, ylabel, zero_line=False):
        bp = ax.boxplot(
            d, tick_labels=l, showfliers=False, patch_artist=True, widths=0.6, vert=True
        )
        for patch, clr in zip(bp["boxes"], c):
            patch.set(facecolor=clr, edgecolor="#333333", linewidth=1.2, alpha=0.85)
        for elem in ["whiskers", "caps"]:
            for line in bp[elem]:
                line.set(color="#555555", linewidth=1.1)
        for med in bp["medians"]:
            med.set(color="#222222", linewidth=1.6)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#aaaaaa", alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        if zero_line:
            ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(ylabel)

    # Signed bias — sorted descending by median
    med_signed = np.array([np.nanmedian(d) for d in data])
    idx_s = np.argsort(med_signed)[::-1]
    l_s = [labels[i] for i in idx_s]
    d_s = [data[i] for i in idx_s]
    c_s = [color_map[lbl] for lbl in l_s]

    log(f"Rendering bias boxplot -> {out_signed.name}")
    fig1, ax1 = plt.subplots(figsize=(8.6, 5), dpi=300)
    _boxplot(ax1, d_s, l_s, c_s, r"Bias (Model $-$ Obs) [$m\,s^{-1}$]", zero_line=True)
    fig1.tight_layout()
    fig1.savefig(out_signed, bbox_inches="tight")
    plt.close(fig1)

    # Absolute bias — sorted descending by median
    abs_data = [np.abs(d) for d in data]
    med_abs = np.array([np.nanmedian(d) for d in abs_data])
    idx_a = np.argsort(med_abs)[::-1]
    l_a = [labels[i] for i in idx_a]
    d_a = [abs_data[i] for i in idx_a]
    c_a = [color_map[lbl] for lbl in l_a]

    log(f"Rendering abs bias boxplot -> {out_abs.name}")
    fig2, ax2 = plt.subplots(figsize=(8.6, 5), dpi=300)
    _boxplot(ax2, d_a, l_a, c_a, r"Absolute Bias $|$Model $-$ Obs$|$ [$m\,s^{-1}$]")
    fig2.tight_layout()
    fig2.savefig(out_abs, bbox_inches="tight")
    plt.close(fig2)


def draw_bias_boxplots_combined(
    asos_bias: Dict[str, pd.DataFrame],
    gs_bias: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """2x2 grid: bias/abs-bias x ASOS/GS, saved as a single figure."""
    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#B07AA1",
        "#76B7B2", "#59A14F", "#EDC948", "#AF7AA1",
    ]
    order = ["ERA5", "WTK", "HRRR", "WTK-LED CONUS", "WTK-LED Climate", "GWA"]
    _tick_wrap = {
        "WTK-LED CONUS": "WTK-LED\nCONUS",
        "WTK-LED Climate": "WTK-LED\nClimate",
    }

    def _prep(bias_frames):
        labels, data = [], []
        for lbl in order:
            dfb = bias_frames.get(lbl)
            if dfb is not None and not dfb.empty and "bias" in dfb.columns:
                x = pd.to_numeric(dfb["bias"], errors="coerce").to_numpy(dtype="float64")
                x = x[np.isfinite(x)]
                if x.size > 0:
                    labels.append(lbl)
                    data.append(x)
        colors = palette[: len(data)]
        color_map = dict(zip(labels, colors))
        return labels, data, color_map

    def _abs_order(labels, data):
        """Return index array sorted by descending median absolute bias."""
        abs_meds = np.array([np.nanmedian(np.abs(d)) for d in data])
        return np.argsort(abs_meds)[::-1]

    def _apply_order(labels, data, color_map, idx, use_abs=False):
        vals = [np.abs(data[i]) for i in idx] if use_abs else [data[i] for i in idx]
        l = [_tick_wrap.get(labels[i], labels[i]) for i in idx]
        c = [color_map[labels[i]] for i in idx]
        return vals, l, c

    def _boxplot(ax, d, l, c, ylabel, zero_line=False):
        bp = ax.boxplot(
            d, tick_labels=l, showfliers=False, patch_artist=True,
            widths=0.6, vert=True,
        )
        for patch, clr in zip(bp["boxes"], c):
            patch.set(facecolor=clr, edgecolor="#333333", linewidth=1.2, alpha=0.85)
        for elem in ["whiskers", "caps"]:
            for line in bp[elem]:
                line.set(color="#555555", linewidth=1.1)
        for med in bp["medians"]:
            med.set(color="#222222", linewidth=1.6)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#aaaaaa", alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        if zero_line:
            ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="both", labelsize=13)

    asos_l, asos_d, asos_cm = _prep(asos_bias)
    gs_l, gs_d, gs_cm = _prep(gs_bias)

    if not asos_d and not gs_d:
        log("No bias data for combined boxplots; skipping.")
        return

    # Sort order driven by absolute bias (shared between signed and abs rows)
    asos_idx = _abs_order(asos_l, asos_d)
    gs_idx = _abs_order(gs_l, gs_d)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300, layout="constrained")

    panel_specs = [
        (axes[0, 0], asos_d, asos_l, asos_cm, asos_idx, False, True,  "(a) Bias (ASOS)"),
        (axes[0, 1], gs_d,   gs_l,   gs_cm,   gs_idx,   False, True,  "(b) Bias (GS)"),
        (axes[1, 0], asos_d, asos_l, asos_cm, asos_idx, True,  False, "(c) Absolute Bias (ASOS)"),
        (axes[1, 1], gs_d,   gs_l,   gs_cm,   gs_idx,   True,  False, "(d) Absolute Bias (GS)"),
    ]

    for ax, data, labels, cmap, idx, use_abs, zero_line, panel_label in panel_specs:
        if not data:
            continue
        d, l, c = _apply_order(labels, data, cmap, idx, use_abs=use_abs)
        ylabel = (
            r"Absolute Bias $|$Model $-$ Obs$|$ [$m\,s^{-1}$]" if use_abs
            else r"Bias (Model $-$ Obs) [$m\,s^{-1}$]"
        )
        _boxplot(ax, d, l, c, ylabel, zero_line=zero_line)
        ax.annotate(
            panel_label, xy=(0.5, 0), xycoords="axes fraction",
            xytext=(0, -46), textcoords="offset points",
            ha="center", va="top", fontsize=18,
        )

    log(f"Rendering combined bias boxplots -> {out_path.name}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _setup_geo_ax(ax, ne_res: str = "50m", conus: bool = True) -> None:
    """Add cartopy basemap features to an existing GeoAxes."""
    ax.add_feature(cfeature.LAND.with_scale(ne_res), facecolor="#FFFFFF", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale(ne_res), facecolor="#cccccc", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale(ne_res), facecolor="#cccccc", edgecolor="none", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale(ne_res), edgecolor="#666666", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale(ne_res), edgecolor="#666666", linewidth=0.5, zorder=2)
    states = NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", ne_res, edgecolor="#999999", facecolor="none")
    ax.add_feature(states, linewidth=0.4, zorder=2)
    if conus:
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())


def draw_spatial_bias_combined(
    asos_bias: Dict[str, pd.DataFrame],
    gs_bias: Dict[str, pd.DataFrame],
    out_path: Path,
    ne_res: str = "50m",
) -> None:
    """2x2 spatial bias map: ERA5 & GWA x ASOS & GS, shared color limits."""
    if not HAS_CARTOPY:
        log("[WARN] Cartopy not available; skipping spatial bias figure.")
        return

    # Collect all bias values for shared symmetric limits
    all_bias = []
    for bdict in [asos_bias, gs_bias]:
        for lbl in ["ERA5", "GWA"]:
            b = bdict.get(lbl)
            if b is not None and not b.empty:
                all_bias.append(b["bias"])
    L = symmetric_bias_limit(all_bias, trim=0.02) if all_bias else 2.0

    panels = [
        (0, 0, asos_bias.get("ERA5"), "o",  18.0, "#444", 0.2, 0.9,  "(a) ERA5 \u2013 ASOS"),
        (0, 1, asos_bias.get("GWA"),  "o",  18.0, "#444", 0.2, 0.9,  "(b) GWA \u2013 ASOS"),
        (1, 0, gs_bias.get("ERA5"),   "D",  42.0, "#111", 0.35, 0.95, "(c) ERA5 \u2013 GS"),
        (1, 1, gs_bias.get("GWA"),    "D",  42.0, "#111", 0.35, 0.95, "(d) GWA \u2013 GS"),
    ]

    proj = ccrs.PlateCarree()
    norm = TwoSlopeNorm(vmin=-L, vcenter=0.0, vmax=L)

    fig, axes = plt.subplots(
        2, 2, figsize=(18, 11), dpi=300,
        subplot_kw={"projection": proj},
    )
    fig.subplots_adjust(wspace=0.05, hspace=-0.05, right=0.88)

    last_sc = None
    for row, col, bdf, marker, ms, edge, lw, alpha, label in panels:
        ax = axes[row, col]
        _setup_geo_ax(ax, ne_res=ne_res, conus=True)

        if bdf is not None and not bdf.empty:
            last_sc = ax.scatter(
                bdf["lon"], bdf["lat"], c=bdf["bias"],
                cmap="RdBu_r", norm=norm, s=ms,
                marker=marker, linewidths=lw, edgecolors=edge,
                alpha=alpha, zorder=3, transform=proj,
            )

        ax.annotate(
            label, xy=(0.5, 0), xycoords="axes fraction",
            xytext=(0, -14), textcoords="offset points",
            ha="center", va="top", fontsize=18,
        )

    # Single colorbar aligned to map edges
    if last_sc is not None:
        top_pos = axes[0, 1].get_position()
        bot_pos = axes[1, 1].get_position()
        cax = fig.add_axes([0.90, bot_pos.y0, 0.02, top_pos.y1 - bot_pos.y0])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label(r"Bias ($m\,s^{-1}$)", fontsize=16)
        cbar.ax.tick_params(labelsize=13)

    log(f"Rendering spatial bias figure -> {out_path.name}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_ws_scatter(
    df_xy: pd.DataFrame,
    title: str,
    out_path: Path,
    xylim_min: float,
    xylim_max: float,
    L_bias: float,
) -> None:
    """Obs vs model scatter colored by bias."""
    if df_xy is None or df_xy.empty:
        log(f"{title}: no data; skipping.")
        return
    log(f"Rendering scatter: {title} -> {out_path.name}")
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=300)

    norm = TwoSlopeNorm(vmin=-L_bias, vcenter=0.0, vmax=L_bias)
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

    ax.set_xlabel(r"Observed mean wind speed ($m\,s^{-1}$)")
    ax.set_ylabel(r"Dataset mean wind speed ($m\,s^{-1}$)")
    ax.grid(True, linestyle="--", linewidth=0.6, color="#aaaaaa", alpha=0.6)

    cbar = plt.colorbar(sc, ax=ax, pad=0.012, fraction=0.046)
    cbar.set_label(r"Bias ($m\,s^{-1}$)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_scatter_combined(
    asos_datasets: Dict[str, pd.DataFrame],
    asos_obs: pd.DataFrame,
    asos_join_cols: List[str],
    gs_datasets: Dict[str, pd.DataFrame],
    gs_obs: pd.DataFrame,
    gs_join_cols: List[str],
    out_path: Path,
) -> None:
    """4x3 scatter grid: 6 ASOS panels (top) + 6 GS panels (bottom).

    Single horizontal colorbar at the top of the figure.
    """
    dataset_order = ["ERA5", "GWA", "HRRR", "WTK", "WTK-LED Climate", "WTK-LED CONUS"]

    # Build all xy data and determine shared limits
    xy_data: list[tuple[str, str, pd.DataFrame]] = []
    all_vals: list[np.ndarray] = []
    all_bias: list[np.ndarray] = []

    for cohort_label, datasets, obs, jcols in [
        ("ASOS", asos_datasets, asos_obs, asos_join_cols),
        ("GS", gs_datasets, gs_obs, gs_join_cols),
    ]:
        for ds_label in dataset_order:
            df_d = datasets.get(ds_label)
            if df_d is None or df_d.empty:
                xy_data.append((ds_label, cohort_label, pd.DataFrame()))
                continue
            xy = _build_xy_bias(df_d, obs, jcols)
            xy_data.append((ds_label, cohort_label, xy))
            if not xy.empty:
                all_vals.extend([xy["x_obs"].to_numpy(), xy["y_model"].to_numpy()])
                all_bias.append(xy["bias"].to_numpy())

    if not all_vals:
        log("[WARN] No scatter data; skipping combined figure.")
        return

    xylim_min, xylim_max = 0, 10
    L_bias = 2.0
    if all_bias:
        pooled = np.concatenate(all_bias)
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size > 0:
            L_bias = float(np.nanpercentile(np.abs(pooled), 98))
            L_bias = max(L_bias, 0.5)

    norm = TwoSlopeNorm(vmin=-L_bias, vcenter=0.0, vmax=L_bias)
    nrows, ncols = 4, 3

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(20, 22), dpi=300,
    )
    fig.subplots_adjust(
        left=0.07, right=0.88, top=0.93, bottom=0.06,
        wspace=0.28, hspace=0.30,
    )

    # Horizontal colorbar at top
    cax = fig.add_axes([0.15, 0.955, 0.70, 0.015])

    last_sc = None
    for idx, (ds_label, cohort_label, xy) in enumerate(xy_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if xy.empty:
            ax.set_visible(False)
            continue

        sc = ax.scatter(
            xy["x_obs"], xy["y_model"], c=xy["bias"],
            cmap="RdBu_r", norm=norm, s=14,
            linewidths=0.2, edgecolors="#333333", alpha=0.8,
        )
        last_sc = sc

        lim = 8 if cohort_label == "ASOS" else xylim_max
        ax.plot(
            [xylim_min, lim], [xylim_min, lim],
            linestyle="--", linewidth=1.0, color="#444444", alpha=0.9,
        )
        ax.set_xlim(xylim_min, lim)
        ax.set_ylim(xylim_min, lim)
        if cohort_label == "ASOS":
            ax.set_xticks([0, 2, 4, 6, 8])
            ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.6, color="#aaaaaa", alpha=0.6)

        if col == 0:
            ax.set_ylabel(r"Dataset mean wind speed ($m\,s^{-1}$)", fontsize=20)
            ax.yaxis.set_label_coords(-0.18, 0.5)
        if row == nrows - 1:
            ax.set_xlabel(r"Observed mean wind speed ($m\,s^{-1}$)", fontsize=20, labelpad=36)
        ax.tick_params(axis="both", labelsize=17)

        panel_letter = chr(ord("a") + idx)
        panel_text = f"({panel_letter}) {ds_label} \u2013 {cohort_label}"
        ax.annotate(
            panel_text, xy=(0.5, 0), xycoords="axes fraction",
            xytext=(0, -28), textcoords="offset points",
            ha="center", va="top", fontsize=20,
        )

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, cax=cax, orientation="horizontal")
        cbar.set_label(r"Bias ($m\,s^{-1}$)", fontsize=20)
        cbar.ax.tick_params(labelsize=17)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

    log(f"Rendering combined scatter figure -> {out_path.name}")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def _render_set(
    datasets: Dict[str, pd.DataFrame],
    df_obs: pd.DataFrame,
    outdir: Path,
    join_cols: List[str],
    trim: float,
    bias_trim: float,
    conus: bool,
    ne_res: str,
    prefix: str,
    gs: bool = False,
    skip_scatter: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Render mean-WS maps, bias maps, and scatter plots for one cohort.

    Returns a dict of ``{label: bias_df}`` for downstream boxplots.
    """
    # Mean WS limits
    mean_series = [d["mean_ws"] for d in datasets.values() if not d.empty]
    if df_obs is not None and not df_obs.empty:
        mean_series.append(df_obs["mean_ws"])
    lo, hi = robust_limits(mean_series, trim=trim) if mean_series else (0.0, 10.0)
    vmin_m, vmax_m = max(0.0, lo), hi
    log(f"[{prefix}] Mean WS robust limits: [{vmin_m:.2f}, {vmax_m:.2f}]")

    marker = "D" if gs else "o"
    ms = 42.0 if gs else 18.0
    edge = "#111" if gs else "#444"
    lw = 0.35 if gs else 0.2
    a = 0.95 if gs else 0.9

    # Mean WS maps
    if df_obs is not None and not df_obs.empty:
        draw_map(
            df_obs,
            f"{prefix}: Obs Mean WS",
            r"Mean wind speed ($m\,s^{-1}$)",
            outdir / f"{prefix.lower()}_obs_mean_ws_map.png",
            vmin_m,
            vmax_m,
            cmap="viridis",
            conus=conus,
            ne_res=ne_res,
            marker=marker,
            ms=ms,
            edgecolors=edge,
            linewidths=lw,
            alpha=a,
        )
    for label, df_d in datasets.items():
        if df_d.empty:
            continue
        tag = label.lower().replace(" ", "_").replace("-", "_")
        draw_map(
            df_d,
            f"{prefix}/{label}: Mean WS",
            r"Mean wind speed ($m\,s^{-1}$)",
            outdir / f"{prefix.lower()}_{tag}_mean_ws_map.png",
            vmin_m,
            vmax_m,
            cmap="viridis",
            conus=conus,
            ne_res=ne_res,
            marker=marker,
            ms=ms,
            edgecolors=edge,
            linewidths=lw,
            alpha=a,
        )

    # Bias maps
    bias_frames: Dict[str, pd.DataFrame] = {}
    bias_series_all: List[pd.Series] = []
    for label, df_d in datasets.items():
        if df_d.empty:
            continue
        b = build_bias_df(df_d, df_obs, label, join_cols)
        if b is not None and not b.empty:
            bias_frames[label] = b
            bias_series_all.append(b["bias"])

    L = (
        symmetric_bias_limit(bias_series_all, trim=bias_trim)
        if bias_series_all
        else 5.0
    )
    log(f"[{prefix}] Bias symmetric limit: +/-{L:.2f}")

    for label, b in bias_frames.items():
        tag = label.lower().replace(" ", "_").replace("-", "_")
        draw_map(
            b,
            f"{prefix} Bias ({label} - Obs)",
            r"Bias ($m\,s^{-1}$)",
            outdir / f"{prefix.lower()}_bias_{tag}_minus_obs_map.png",
            -L,
            L,
            cmap="RdBu_r",
            diverging=True,
            conus=conus,
            value_col="bias",
            ne_res=ne_res,
            marker=marker,
            ms=ms,
            edgecolors=edge,
            linewidths=lw,
            alpha=a,
        )

    # Scatter plots
    if not skip_scatter and df_obs is not None and not df_obs.empty:
        for label, df_d in datasets.items():
            if df_d.empty:
                continue
            xy = _build_xy_bias(df_d, df_obs, join_cols)
            tag = label.lower().replace(" ", "_").replace("-", "_")
            draw_ws_scatter(
                xy,
                f"{prefix}: {label} vs Obs",
                outdir / f"scatter_{prefix.lower()}_{tag}_vs_obs.png",
                vmin_m,
                vmax_m,
                L,
            )

    return bias_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Maps of mean wind speed and bias for ASOS, "
            "Gold Standard, and overlays (+GWA)."
        )
    )

    # ASOS cohort
    p.add_argument("--obs", type=Path, required=True, help="ASOS observation quantile CSV")
    p.add_argument("--era5", type=Path, required=True, help="ERA5 quantile CSV")
    p.add_argument("--wtk", type=Path, required=True, help="WTK quantile CSV")
    p.add_argument("--hrrr", type=Path, required=True, help="HRRR quantile CSV")
    p.add_argument("--ledc", type=Path, required=True, help="WTK-LED CONUS quantile CSV")
    p.add_argument("--ledclim", type=Path, required=True, help="WTK-LED Climate quantile CSV")

    # GS cohort
    p.add_argument("--gs-obs", type=Path, required=True, help="GS observation quantile CSV")
    p.add_argument("--gs-era5", type=Path, required=True, help="GS ERA5 quantile CSV")
    p.add_argument("--gs-wtk", type=Path, required=True, help="GS WTK quantile CSV")
    p.add_argument("--gs-hrrr", type=Path, required=True, help="GS HRRR quantile CSV")
    p.add_argument("--gs-ledc", type=Path, required=True, help="GS WTK-LED CONUS quantile CSV")
    p.add_argument("--gs-ledclim", type=Path, required=True, help="GS WTK-LED Climate quantile CSV")

    # Optional GWA
    p.add_argument("--gwa", type=Path, default=None, help="GWA sites CSV (ASOS)")
    p.add_argument("--gs-gwa", type=Path, default=None, help="GWA sites CSV (GS)")

    # Output & display
    p.add_argument("--outdir", type=Path, default=Path("."), help="Output directory")
    p.add_argument("--conus", dest="conus", action="store_true", default=True, help="Crop to CONUS extent")
    p.add_argument("--no-conus", dest="conus", action="store_false", help="Do not crop maps to CONUS extent")
    p.add_argument("--ne-res", type=str, default="50m", choices=["110m", "50m", "10m"])
    p.add_argument("--trim", type=float, default=0.02, help="Trim for robust limits")

    # Skip flags
    p.add_argument("--skip-scatter", action="store_true", help="Skip scatter plots")
    p.add_argument("--skip-overlay", action="store_true", help="Skip overlay maps")
    p.add_argument("--skip-boxplot", action="store_true", help="Skip bias boxplots")

    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load ASOS datasets ----
    df_obs = load_quantile_file(args.obs, "Observations")
    df_era5 = load_quantile_file(args.era5, "ERA5")
    df_wtk = load_quantile_file(args.wtk, "WTK")
    df_hrrr = load_quantile_file(args.hrrr, "HRRR")
    df_ledc = load_quantile_file(args.ledc, "WTK-LED CONUS")
    df_ledclim = load_quantile_file(args.ledclim, "WTK-LED Climate")
    df_gwa = load_gwa_file(args.gwa, "ASOS")

    # ---- Load GS datasets ----
    df_gs_obs = load_quantile_file(args.gs_obs, "GS Observations")
    df_gs_era5 = load_quantile_file(args.gs_era5, "GS ERA5")
    df_gs_wtk = load_quantile_file(args.gs_wtk, "GS WTK")
    df_gs_hrrr = load_quantile_file(args.gs_hrrr, "GS HRRR")
    df_gs_ledc = load_quantile_file(args.gs_ledc, "GS WTK-LED CONUS")
    df_gs_ledclim = load_quantile_file(args.gs_ledclim, "GS WTK-LED Climate")
    df_gs_gwa = load_gwa_file(args.gs_gwa, "GS")

    # ---- A) ASOS set (outlier-filtered using ERA5-Obs bias) ----
    keep_ids = filter_by_bias(df_obs, df_era5, bias_trim=args.trim)
    log(f"[ASOS] Keeping {len(keep_ids)} stations after ERA5-bias filtering.")

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["station_id"].isin(keep_ids)].copy()

    asos_datasets: Dict[str, pd.DataFrame] = {
        "ERA5": keep(df_era5),
        "WTK": keep(df_wtk),
        "HRRR": keep(df_hrrr),
        "WTK-LED CONUS": keep(df_ledc),
        "WTK-LED Climate": keep(df_ledclim),
    }
    if not df_gwa.empty:
        asos_datasets["GWA"] = keep(df_gwa)

    asos_bias = _render_set(
        asos_datasets,
        keep(df_obs),
        args.outdir,
        join_cols=["station_id"],
        trim=args.trim,
        bias_trim=args.trim,
        conus=args.conus,
        ne_res=args.ne_res,
        prefix="ASOS",
        skip_scatter=args.skip_scatter,
    )

    if not args.skip_boxplot:
        draw_bias_boxplots(
            asos_bias,
            args.outdir / "bias_boxplot.png",
            args.outdir / "abs_bias_boxplot.png",
        )

    # ---- B) Gold Standard set (no outlier removal) ----
    gs_datasets: Dict[str, pd.DataFrame] = {
        "ERA5": df_gs_era5,
        "WTK": df_gs_wtk,
        "HRRR": df_gs_hrrr,
        "WTK-LED CONUS": df_gs_ledc,
        "WTK-LED Climate": df_gs_ledclim,
    }
    if not df_gs_gwa.empty:
        gs_datasets["GWA"] = df_gs_gwa

    gs_bias = _render_set(
        gs_datasets,
        df_gs_obs,
        args.outdir,
        join_cols=["station_id", "height_m"],
        trim=args.trim,
        bias_trim=args.trim,
        conus=args.conus,
        ne_res=args.ne_res,
        prefix="GS",
        gs=True,
        skip_scatter=args.skip_scatter,
    )

    if not args.skip_boxplot:
        draw_bias_boxplots(
            gs_bias,
            args.outdir / "gs_bias_boxplot.png",
            args.outdir / "gs_abs_bias_boxplot.png",
        )
        draw_bias_boxplots_combined(
            asos_bias, gs_bias,
            args.outdir / "dataset_bias_boxplots.pdf",
        )

    # ---- Combined spatial bias figure (ERA5 + GWA x ASOS + GS) ----
    draw_spatial_bias_combined(
        asos_bias, gs_bias,
        args.outdir / "spatial_bias.pdf",
        ne_res=args.ne_res,
    )

    # ---- Combined scatter figure (all datasets x ASOS + GS) ----
    if not args.skip_scatter:
        draw_scatter_combined(
            asos_datasets, keep(df_obs), ["station_id"],
            gs_datasets, df_gs_obs, ["station_id", "height_m"],
            args.outdir / "bias_scatterplots.pdf",
        )

    # ---- C) Overlay maps ----
    if not args.skip_overlay:
        # Pool limits across both sets
        all_means = []
        for d in list(asos_datasets.values()) + list(gs_datasets.values()) + [keep(df_obs), df_gs_obs]:
            if d is not None and not d.empty:
                all_means.append(d["mean_ws"])
        lo_ov, hi_ov = robust_limits(all_means, trim=args.trim) if all_means else (0.0, 1.0)
        vmin_ov, vmax_ov = max(0.0, lo_ov), hi_ov

        # Pooled bias limits
        all_bias = []
        for bdict in [asos_bias, gs_bias]:
            for b in bdict.values():
                if not b.empty:
                    all_bias.append(b["bias"])
        L_ov = symmetric_bias_limit(all_bias, trim=args.trim) if all_bias else 5.0

        # Overlay mean WS
        asos_obs_filt = keep(df_obs)
        if not asos_obs_filt.empty and not df_gs_obs.empty:
            draw_overlay_map(
                asos_obs_filt,
                df_gs_obs,
                "Overlay: Obs Mean WS",
                r"Mean wind speed ($m\,s^{-1}$)",
                args.outdir / "overlay_obs_mean_ws_map.png",
                vmin_ov,
                vmax_ov,
                "viridis",
                False,
                args.conus,
                "mean_ws",
                args.ne_res,
            )
        ds_tags = ["ERA5", "WTK", "HRRR", "WTK-LED CONUS", "WTK-LED Climate"]
        if not df_gwa.empty and not df_gs_gwa.empty:
            ds_tags.append("GWA")
        for tag in ds_tags:
            d_a = asos_datasets.get(tag)
            d_g = gs_datasets.get(tag)
            if d_a is not None and not d_a.empty and d_g is not None and not d_g.empty:
                ftag = tag.lower().replace(" ", "_").replace("-", "_")
                draw_overlay_map(
                    d_a,
                    d_g,
                    f"Overlay: {tag} Mean WS",
                    r"Mean wind speed ($m\,s^{-1}$)",
                    args.outdir / f"overlay_{ftag}_mean_ws_map.png",
                    vmin_ov,
                    vmax_ov,
                    "viridis",
                    False,
                    args.conus,
                    "mean_ws",
                    args.ne_res,
                )

        # Overlay bias maps
        for tag in ds_tags:
            b_a = asos_bias.get(tag)
            b_g = gs_bias.get(tag)
            if (
                b_a is not None
                and not b_a.empty
                and b_g is not None
                and not b_g.empty
            ):
                ftag = tag.lower().replace(" ", "_").replace("-", "_")
                draw_overlay_map(
                    b_a,
                    b_g,
                    f"Overlay: Bias ({tag} - Obs)",
                    r"Bias ($m\,s^{-1}$)",
                    args.outdir / f"overlay_bias_{ftag}_minus_obs_map.png",
                    -L_ov,
                    L_ov,
                    "RdBu_r",
                    True,
                    args.conus,
                    "bias",
                    args.ne_res,
                )

    log(f"Done. Outputs in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
