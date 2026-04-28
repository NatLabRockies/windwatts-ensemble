#!/usr/bin/env python3
"""
Make per-height maps from merged per-dataset quantile files (HRRR, WTK, WTK-LED),
from a single merged ML predictions file, and from GWA values precomputed at the
ML inference heights (via inference_table_w_gwa.*).

Also produces per-height DIFF maps:  (ML - dataset)
  * red   : ML predicts LOWER than dataset (negative)
  * white : zero difference
  * blue  : ML predicts HIGHER than dataset (positive)
Diff color limits are constant across all diff maps (global symmetric bounds).

Sequential (mean) color ramp (fixed 0->10+ m/s):
    0 m/s  -> white (#FFFFFF)
    ->      -> blue  (#5A8DC1)
    ->      -> yellow(#F7E873)
    ->      -> red   (#C33732)
    >=10 m/s-> purple(#8F3A63)   (colorbar extend='max' shows 10+)

Outputs (to --out-dir):
  - map_<prefix>_mean_<height>m.(png|pdf)   for datasets
  - map_pred_mean_<height>m.(png|pdf)       for predictions
  - map_gwa_mean_<height>m.(png|pdf)        for GWA (using gwa_interp)
  - map_diff_pred_minus_<prefix>_<height>m.(png|pdf)  for each dataset diff
  - map_diff_pred_minus_gwa_<height>m.(png|pdf)       for GWA diff

Notes
-----
* GWA is read from an inference table that already contains the vertically
  interpolated GWA at the target hub heights (column: gwa_interp). We
  deduplicate to one row per (lat,lon,height_m) before mapping.
* If --global-scale is requested, percentiles are computed across ALL sources
  (datasets, predictions, and GWA) and applied to every mean map.
* Diff limits are computed once across ALL diffs (unless --diff-limit is set).
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from wem.utils.logging import log
from wem.utils.quantiles import mean_from_quantiles
from wem.utils.plotting import make_custom_cmap, make_diff_cmap, wrap_lon180, mask_points_to_us

QCOLS = [f"q{p:03d}" for p in range(101)]

# ---------- custom 0->10+ colormap for MEAN ----------
CUSTOM_CMAP_MEAN = make_custom_cmap()

# ---------- diverging red-white-blue colormap for DIFF ----------
CUSTOM_CMAP_DIFF = make_diff_cmap()

# ---------- Input discovery & reading (datasets) ----------
def list_height_files(in_dir: Path, prefix: str) -> List[Tuple[int, Path]]:
    pats = [f"{prefix}_quantiles_*m.csv", f"{prefix}_quantiles_*m.parquet", f"{prefix}_quantiles_*m.pq"]
    files: List[Path] = []
    for pat in pats:
        files += sorted(in_dir.glob(pat))
    out: List[Tuple[int, Path]] = []
    rx = re.compile(rf"{re.escape(prefix)}_quantiles_(\d+)m\.(?:csv|parquet|pq)$")
    for fp in files:
        m = rx.search(fp.name)
        if m:
            out.append((int(m.group(1)), fp))
    return sorted(out, key=lambda t: t[0])

def read_height_file(fp: Path, max_points: Optional[int] = None) -> pd.DataFrame:
    usecols = ["lat", "lon", "height_m"] + QCOLS
    if fp.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(fp, columns=None)
        df = df[usecols].copy()
    else:
        try:
            df = pd.read_csv(fp, usecols=lambda c: (c in usecols))
        except Exception:
            df = pd.read_csv(fp)[usecols]
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)
    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).reset_index(drop=True)
    return df

# ---------- Input reading (predictions file) ----------
def read_predictions(pred_path: Path, max_points: Optional[int] = None) -> pd.DataFrame:
    usecols = ["latitude", "longitude", "height_m"] + QCOLS
    if pred_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(pred_path, columns=None)
        df = df[[c for c in usecols if c in df.columns]].copy()
    else:
        try:
            df = pd.read_csv(pred_path, usecols=lambda c: (c in usecols))
        except Exception:
            df = pd.read_csv(pred_path)
            df = df[[c for c in usecols if c in df.columns]]
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Predictions file must have 'latitude' and 'longitude' columns.")
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)
    for qc in QCOLS:
        if qc not in df.columns:
            df[qc] = np.nan
    df = df[["lat", "lon", "height_m"] + QCOLS].copy()
    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).reset_index(drop=True)
    return df

def unique_prediction_heights(df_pred: pd.DataFrame) -> List[int]:
    hs = pd.to_numeric(df_pred["height_m"], errors="coerce").dropna().unique()
    return sorted(int(round(float(h))) for h in hs)

# ---------- Input reading (GWA inference table) ----------
def read_gwa_table(gwa_path: Path, max_points: Optional[int] = None) -> pd.DataFrame:
    """
    Read inference_table_w_gwa.* and return a table with columns:
      lat, lon, height_m, gwa_interp
    Deduplicated to one row per (lat,lon,height_m).
    """
    usecols = ["lat", "lon", "height_m", "gwa_interp"]
    if gwa_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(gwa_path, columns=None)
        df = df[[c for c in usecols if c in df.columns]].copy()
    else:
        try:
            df = pd.read_csv(gwa_path, usecols=lambda c: (c in usecols))
        except Exception:
            df = pd.read_csv(gwa_path)
            df = df[[c for c in usecols if c in df.columns]]
    for c in ["lat", "lon", "height_m", "gwa_interp"]:
        if c not in df.columns:
            raise ValueError(f"GWA table missing column: {c}")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["height_m"] = pd.to_numeric(df["height_m"], errors="coerce")
    df["gwa_interp"] = pd.to_numeric(df["gwa_interp"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "height_m"]).reset_index(drop=True)
    df = df.sort_values(["lat", "lon", "height_m"]).drop_duplicates(["lat", "lon", "height_m"])
    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    df = df.reset_index(drop=True)
    return df

def unique_gwa_heights(df_gwa: pd.DataFrame) -> List[int]:
    hs = pd.to_numeric(df_gwa["height_m"], errors="coerce").dropna().unique()
    return sorted(int(round(float(h))) for h in hs)

def plot_mean_height_us_only(
    df: pd.DataFrame,
    values: np.ndarray,     # mean speeds (m/s)
    out_fp: Path,
    vmin: float,
    vmax: float,
    s: float,
    alpha: float,
    dpi: int,
    extent: Tuple[float,float,float,float],
    grid_dx: float,
    grid_dy: float,
    ocean_face: str,
    us_shapefile: Optional[Path],
    verbose_counts: bool = True,
) -> None:
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 6.8))
    ax = plt.axes(projection=pc)
    ax.set_extent(extent, crs=pc)

    lon_raw = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = wrap_lon180(lon_raw)

    # 1) extent filter
    e_lon_min, e_lon_max, e_lat_min, e_lat_max = extent
    in_ext = (lon >= e_lon_min) & (lon <= e_lon_max) & (lat >= e_lat_min) & (lat <= e_lat_max)

    # 2) U.S. mask
    in_us = mask_points_to_us(lon[in_ext], lat[in_ext], us_shapefile=us_shapefile)

    if verbose_counts:
        log(f"Points total={len(df):,}  in_extent={int(in_ext.sum()):,}  in_US={int(in_us.sum()):,}")

    if not in_us.any():
        fig.text(0.5, 0.55, "No points within U.S. mask", ha="center", va="center", fontsize=12)
        fig.text(0.5, 0.45,
                 "If this seems wrong, check extent bounds, lon/lat columns, or provide --us-shapefile.",
                 ha="center", va="center", fontsize=9)
        fig.savefig(out_fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    lonp = lon[in_ext][in_us]
    latp = lat[in_ext][in_us]
    vp = values[in_ext][in_us]

    mappable = ax.scatter(
        lonp, latp,
        c=vp, s=s, marker='s',
        cmap=CUSTOM_CMAP_MEAN, vmin=vmin, vmax=vmax,
        transform=pc, linewidths=0, alpha=alpha, zorder=4,
    )

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=ocean_face, edgecolor="none", zorder=5)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=ocean_face, edgecolor="none", zorder=6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, zorder=7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=7)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, zorder=7)

    import matplotlib.ticker as mticker
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
    gl.xlocator = mticker.MultipleLocator(base=grid_dx)
    gl.ylocator = mticker.MultipleLocator(base=grid_dy)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    cb = plt.colorbar(mappable, ax=ax, orientation="vertical", pad=0.02, shrink=0.9, extend="max")
    cb.set_label(r"Mean wind speed ($m\,s^{-1}$)")
    try:
        cb.set_ticks(np.linspace(vmin, vmax, int(max(2, round(vmax - vmin))) + 1))
    except Exception:
        pass

    ax.set_title(out_fp.stem.replace("_", " "), fontsize=11)
    fig.savefig(out_fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_diff_height_us_only(
    df: pd.DataFrame,
    diffs: np.ndarray,       # ML - dataset (m/s)
    out_fp: Path,
    L: float,                # symmetric limit (color bounds = [-L, +L])
    s: float,
    alpha: float,
    dpi: int,
    extent: Tuple[float,float,float,float],
    grid_dx: float,
    grid_dy: float,
    ocean_face: str,
    us_shapefile: Optional[Path],
    verbose_counts: bool = True,
) -> None:
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 6.8))
    ax = plt.axes(projection=pc)
    ax.set_extent(extent, crs=pc)

    lon_raw = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = wrap_lon180(lon_raw)

    # 1) extent filter
    e_lon_min, e_lon_max, e_lat_min, e_lat_max = extent
    in_ext = (lon >= e_lon_min) & (lon <= e_lon_max) & (lat >= e_lat_min) & (lat <= e_lat_max)

    # 2) U.S. mask
    in_us = mask_points_to_us(lon[in_ext], lat[in_ext], us_shapefile=us_shapefile)

    if verbose_counts:
        log(f"Points total={len(df):,}  in_extent={int(in_ext.sum()):,}  in_US={int(in_us.sum()):,}")

    if not in_us.any():
        fig.text(0.5, 0.55, "No points within U.S. mask", ha="center", va="center", fontsize=12)
        fig.text(0.5, 0.45,
                 "If this seems wrong, check extent bounds, lon/lat columns, or provide --us-shapefile.",
                 ha="center", va="center", fontsize=9)
        fig.savefig(out_fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    lonp = lon[in_ext][in_us]
    latp = lat[in_ext][in_us]
    dp = diffs[in_ext][in_us]

    norm = TwoSlopeNorm(vmin=-L, vcenter=0.0, vmax=+L)

    mappable = ax.scatter(
        lonp, latp,
        c=dp, s=s, marker='s',
        cmap=CUSTOM_CMAP_DIFF, norm=norm,
        transform=pc, linewidths=0, alpha=alpha, zorder=4,
    )

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=ocean_face, edgecolor="none", zorder=5)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=ocean_face, edgecolor="none", zorder=6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, zorder=7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=7)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, zorder=7)

    import matplotlib.ticker as mticker
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
    gl.xlocator = mticker.MultipleLocator(base=grid_dx)
    gl.ylocator = mticker.MultipleLocator(base=grid_dy)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    cb = plt.colorbar(mappable, ax=ax, orientation="vertical", pad=0.02, shrink=0.9)
    cb.set_label(r"ML $-$ dataset ($m\,s^{-1}$)")
    cb.set_ticks(np.linspace(-L, +L, 9))

    ax.set_title(out_fp.stem.replace("_", " "), fontsize=11)
    fig.savefig(out_fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- Global scale helpers for MEAN ----------
def collect_means_for_files(files: List[Tuple[int, Path]], max_points: Optional[int]) -> np.ndarray:
    vals: List[np.ndarray] = []
    for _, fp in files:
        df = read_height_file(fp, max_points=max_points)
        vals.append(mean_from_quantiles(df))
    return (np.concatenate(vals) if vals else np.array([], dtype="float32"))

def collect_means_for_predictions(df_pred: pd.DataFrame, max_points: Optional[int]) -> np.ndarray:
    vals: List[np.ndarray] = []
    for _, g in df_pred.groupby(df_pred["height_m"]):
        if max_points is not None and len(g) > max_points:
            g = g.sample(n=max_points, random_state=42)
        vals.append(mean_from_quantiles(g))
    return (np.concatenate(vals) if vals else np.array([], dtype="float32"))

def collect_values_for_gwa(df_gwa: pd.DataFrame, max_points: Optional[int]) -> np.ndarray:
    vals: List[np.ndarray] = []
    for _, g in df_gwa.groupby(df_gwa["height_m"]):
        if max_points is not None and len(g) > max_points:
            g = g.sample(n=max_points, random_state=42)
        v = pd.to_numeric(g["gwa_interp"], errors="coerce").to_numpy(dtype="float32")
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    return (np.concatenate(vals) if vals else np.array([], dtype="float32"))

# ---------- helpers for DIFF preparation ----------
def round_coords(df: pd.DataFrame, prec: int) -> pd.DataFrame:
    out = df.copy()
    out["lat_r"] = out["lat"].round(prec)
    out["lon_r"] = out["lon"].round(prec)
    return out

def build_means_df(df: pd.DataFrame, height: int, coord_prec: int) -> pd.DataFrame:
    """Return lat, lon, mean_ws for a single height from a quantile-based dataset."""
    sub = df.loc[(df["height_m"].round(6) == float(height))].copy()
    if sub.empty:
        return sub
    means = mean_from_quantiles(sub)
    sub = sub.assign(mean_ws=means)[["lat", "lon", "mean_ws"]]
    sub = round_coords(sub, coord_prec)
    return sub

def build_pred_means_df(df_pred: pd.DataFrame, height: int, coord_prec: int) -> pd.DataFrame:
    sub = df_pred.loc[(df_pred["height_m"].round(6) == float(height))].copy()
    if sub.empty:
        return sub
    means = mean_from_quantiles(sub)
    sub = sub.assign(mean_ws=means)[["lat", "lon", "mean_ws"]]
    sub = round_coords(sub, coord_prec)
    return sub

def build_gwa_df(df_gwa: pd.DataFrame, height: int, coord_prec: int) -> pd.DataFrame:
    sub = df_gwa.loc[(df_gwa["height_m"].round(6) == float(height))].copy()
    if sub.empty:
        return sub
    sub = sub.rename(columns={"gwa_interp": "mean_ws"})[["lat", "lon", "mean_ws"]]
    sub = round_coords(sub, coord_prec)
    return sub

def merge_for_diff(df_pred_h: pd.DataFrame, df_ds_h: pd.DataFrame) -> pd.DataFrame:
    if df_pred_h.empty or df_ds_h.empty:
        return pd.DataFrame(columns=["lat","lon","pred","ds"])
    m = pd.merge(
        df_pred_h, df_ds_h,
        on=["lat_r","lon_r"], how="inner", suffixes=("_pred","_ds")
    )
    if m.empty:
        return m
    # Use original (unrounded) coordinates from prediction side for plotting
    out = pd.DataFrame({
        "lat": m["lat_pred"].to_numpy(),
        "lon": m["lon_pred"].to_numpy(),
        "pred": m["mean_ws_pred"].to_numpy(),
        "ds":   m["mean_ws_ds"].to_numpy(),
    })
    return out

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Create per-height mean and diff maps (datasets, ML, GWA).")
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory with <prefix>_quantiles_*m.(csv|parquet|pq)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write map images")
    ap.add_argument("--prefix", type=str, default="hrrr,wtk,wtk_led",
                    help="Comma-separated file prefixes (e.g., hrrr,wtk,wtk_led).")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Path to merged predictions file (site_quantiles_predicted.csv or .parquet).")
    ap.add_argument("--gwa", type=Path, default=None,
                    help="Path to inference_table_w_gwa.(csv|parquet) with 'gwa_interp'.")
    ap.add_argument("--format", type=str, choices=["png", "pdf"], default="png", help="Output image format")
    ap.add_argument("--dpi", type=int, default=160, help="Image DPI")
    ap.add_argument("--point-size", type=float, default=6.0, help="Scatter square size")
    ap.add_argument("--alpha", type=float, default=0.9, help="Square alpha")
    ap.add_argument("--max-points", type=int, default=None, help="Optional subsample cap per height (for scale and plotting)")
    ap.add_argument("--global-scale", action="store_true",
                    help="Compute ONE vmin/vmax for mean maps across ALL sources from percentiles.")
    ap.add_argument("--scale-pct", type=str, default="1,99", help="Percentiles for global mean-map scale if --global-scale")
    ap.add_argument("--vmin", type=float, default=0.0, help="Mean map color minimum (default 0)")
    ap.add_argument("--vmax", type=float, default=10.0, help="Mean map color maximum (default 10)")
    ap.add_argument("--extent", type=float, nargs=4, default=[-125, -66, 24, 50],
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    help="Map extent in degrees (Plate Carree). Default: CONUS.")
    ap.add_argument("--grid-dx", type=float, default=5.0, help="Longitude grid spacing (deg)")
    ap.add_argument("--grid-dy", type=float, default=5.0, help="Latitude grid spacing (deg)")
    ap.add_argument("--ocean-color", type=str, default="white", help="Fill color for oceans/lakes")
    ap.add_argument("--us-shapefile", type=Path, default=None,
                    help="Path to a local U.S. polygon file (e.g., Natural Earth admin_0_countries.shp or a U.S.-only GeoJSON).")
    # Diff controls
    ap.add_argument("--no-diff", action="store_true", help="Disable generating diff maps.")
    ap.add_argument("--diff-limit", type=float, default=None,
                    help="Set symmetric diff color bound (L) so limits are [-L,+L] for all diff maps. If omitted, computed from percentiles.")
    ap.add_argument("--diff-pct", type=str, default="2,98",
                    help="Percentiles to derive symmetric diff bound when --diff-limit not given (e.g., '2,98').")
    ap.add_argument("--coord-precision", type=int, default=6,
                    help="Decimals to round lat/lon before joining ML with datasets/GWA for diffs.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Gather dataset files -----
    prefixes = [p.strip() for p in args.prefix.split(",") if p.strip()]
    prefix_to_files: Dict[str, List[Tuple[int, Path]]] = {}
    all_files: List[Tuple[int, Path]] = []
    for prefix in prefixes:
        files = list_height_files(args.in_dir, prefix=prefix)
        if files:
            prefix_to_files[prefix] = files
            all_files.extend(files)
        else:
            log(f"[{prefix}] No files like '{prefix}_quantiles_*m.(csv|parquet|pq)' in {args.in_dir}; skipping.")

    # ----- Load predictions (optional) -----
    df_pred: Optional[pd.DataFrame] = None
    pred_heights: List[int] = []
    if args.predictions is not None:
        if not args.predictions.exists():
            raise SystemExit(f"Predictions file not found: {args.predictions}")
        log(f"[pred] Reading predictions: {args.predictions.name}")
        df_pred = read_predictions(args.predictions)
        if df_pred.empty:
            log("[pred] WARNING: predictions file is empty after cleaning; skipping.")
            df_pred = None
        else:
            pred_heights = unique_prediction_heights(df_pred)
            log(f"[pred] Heights found: {pred_heights}")

    # ----- Load GWA (optional) -----
    df_gwa: Optional[pd.DataFrame] = None
    gwa_heights: List[int] = []
    if args.gwa is not None:
        if not args.gwa.exists():
            raise SystemExit(f"GWA table not found: {args.gwa}")
        log(f"[gwa] Reading GWA inference table: {args.gwa.name}")
        df_gwa = read_gwa_table(args.gwa)
        if df_gwa.empty:
            log("[gwa] WARNING: GWA table is empty after cleaning; skipping.")
            df_gwa = None
        else:
            gwa_heights = unique_gwa_heights(df_gwa)
            log(f"[gwa] Heights found: {gwa_heights}")

    if not all_files and df_pred is None and df_gwa is None:
        raise SystemExit("No dataset files, no predictions, and no GWA provided -- nothing to do.")

    # ----- Parse percentile bounds for MEAN maps -----
    try:
        lo_pct, hi_pct = (float(x) for x in args.scale_pct.split(","))
    except Exception:
        lo_pct, hi_pct = 1.0, 99.0

    # ----- Compute ONE global mean color scale if requested -----
    if args.global_scale and (args.vmin is None or args.vmax is None):
        means_list: List[np.ndarray] = []
        if all_files:
            means_ds = collect_means_for_files(all_files, args.max_points)
            if means_ds.size:
                means_list.append(means_ds)
        if df_pred is not None and not df_pred.empty:
            means_pr = collect_means_for_predictions(df_pred, args.max_points)
            if means_pr.size:
                means_list.append(means_pr)
        if df_gwa is not None and not df_gwa.empty:
            vals_gwa = collect_values_for_gwa(df_gwa, args.max_points)
            if vals_gwa.size:
                means_list.append(vals_gwa)
        allv = np.concatenate(means_list) if means_list else np.array([], dtype="float32")
        if allv.size == 0:
            vmin_g, vmax_g = float(args.vmin), float(args.vmax)
        else:
            vmin_g = float(np.nanpercentile(allv, lo_pct))
            vmax_g = float(np.nanpercentile(allv, hi_pct))
        log(f"[GLOBAL MEAN] Color scale from percentiles {lo_pct},{hi_pct}: vmin={vmin_g:.3f}, vmax={vmax_g:.3f}")
    else:
        vmin_g = float(args.vmin)
        vmax_g = float(args.vmax)

    # ====================== MEAN MAPS ======================
    # Dataset files
    for prefix, files in prefix_to_files.items():
        for z, fp in files:
            log(f"[{prefix}] Reading {fp.name} ...")
            df = read_height_file(fp, max_points=args.max_points)
            values = mean_from_quantiles(df)
            vmin, vmax = (vmin_g, vmax_g) if args.global_scale else (float(args.vmin), float(args.vmax))
            out_fp = args.out_dir / f"map_{prefix}_mean_{z:03d}m.{args.format}"
            plot_mean_height_us_only(
                df, values, out_fp,
                vmin=vmin, vmax=vmax,
                s=args.point_size, alpha=args.alpha, dpi=args.dpi,
                extent=tuple(args.extent), grid_dx=args.grid_dx, grid_dy=args.grid_dy,
                ocean_face=args.ocean_color,
                us_shapefile=args.us_shapefile,
                verbose_counts=True,
            )
            log(f"[{prefix}] Wrote {out_fp.name}")

    # ML predictions
    if df_pred is not None and not df_pred.empty:
        for z in pred_heights:
            sub = df_pred.loc[(df_pred["height_m"].round(6) == float(z))].copy()
            if sub.empty:
                continue
            if args.max_points is not None and len(sub) > args.max_points:
                sub = sub.sample(n=args.max_points, random_state=42)
            values = mean_from_quantiles(sub)
            vmin, vmax = (vmin_g, vmax_g) if args.global_scale else (float(args.vmin), float(args.vmax))
            out_fp = args.out_dir / f"map_pred_mean_{z:03d}m.{args.format}"
            plot_mean_height_us_only(
                sub.rename(columns={"lat":"lat","lon":"lon"}), values, out_fp,
                vmin=vmin, vmax=vmax,
                s=args.point_size, alpha=args.alpha, dpi=args.dpi,
                extent=tuple(args.extent), grid_dx=args.grid_dx, grid_dy=args.grid_dy,
                ocean_face=args.ocean_color,
                us_shapefile=args.us_shapefile,
                verbose_counts=True,
            )
            log(f"[pred] Wrote {out_fp.name}")

    # GWA
    if df_gwa is not None and not df_gwa.empty:
        for z in gwa_heights:
            sub = df_gwa.loc[(df_gwa["height_m"].round(6) == float(z))].copy()
            if sub.empty:
                continue
            if args.max_points is not None and len(sub) > args.max_points:
                sub = sub.sample(n=args.max_points, random_state=42)
            values = pd.to_numeric(sub["gwa_interp"], errors="coerce").to_numpy(dtype="float32")
            vmin, vmax = (vmin_g, vmax_g) if args.global_scale else (float(args.vmin), float(args.vmax))
            out_fp = args.out_dir / f"map_gwa_mean_{z:03d}m.{args.format}"
            sub_plot = sub[["lat", "lon"]].copy()
            plot_mean_height_us_only(
                sub_plot.assign(lat=sub["lat"], lon=sub["lon"]),
                values, out_fp,
                vmin=vmin, vmax=vmax,
                s=args.point_size, alpha=args.alpha, dpi=args.dpi,
                extent=tuple(args.extent), grid_dx=args.grid_dx, grid_dy=args.grid_dy,
                ocean_face=args.ocean_color,
                us_shapefile=args.us_shapefile,
                verbose_counts=True,
            )
            log(f"[gwa] Wrote {out_fp.name}")

    # ====================== DIFF MAPS (ML - dataset) ======================
    if (not args.no_diff) and (df_pred is not None and not df_pred.empty):
        # Build a dict of per-height prediction means (rounded coords)
        pred_by_h: Dict[int, pd.DataFrame] = {}
        for z in pred_heights:
            pred_by_h[z] = build_pred_means_df(df_pred, z, args.coord_precision)

        # Collect all diffs to compute a single symmetric bound L
        diffs_all: List[np.ndarray] = []

        # Dataset diffs
        for prefix, files in prefix_to_files.items():
            for z, fp in files:
                df_ds_full = read_height_file(fp, max_points=None)  # Don't subsample for diff calcs
                ds_h = build_means_df(df_ds_full, z, args.coord_precision)
                pr_h = pred_by_h.get(z, pd.DataFrame())
                m = merge_for_diff(pr_h, ds_h)
                if not m.empty:
                    diffs_all.append((m["pred"] - m["ds"]).to_numpy(dtype="float64"))

        # GWA diffs
        if df_gwa is not None and not df_gwa.empty:
            for z in gwa_heights:
                gwa_h = build_gwa_df(df_gwa, z, args.coord_precision)
                pr_h = pred_by_h.get(z, pd.DataFrame())
                m = merge_for_diff(pr_h, gwa_h)
                if not m.empty:
                    diffs_all.append((m["pred"] - m["ds"]).to_numpy(dtype="float64"))

        # Determine symmetric diff limit L
        if args.diff_limit is not None:
            L = float(args.diff_limit)
            log(f"[DIFF] Using user-specified symmetric limit L = {L:.3f}")
        else:
            if diffs_all:
                diffs_concat = np.concatenate(diffs_all)
                diffs_concat = diffs_concat[np.isfinite(diffs_concat)]
                try:
                    lo_pct, hi_pct = (float(x) for x in args.diff_pct.split(","))
                except Exception:
                    lo_pct, hi_pct = 2.0, 98.0
                lo = float(np.nanpercentile(diffs_concat, lo_pct))
                hi = float(np.nanpercentile(diffs_concat, hi_pct))
                L = max(abs(lo), abs(hi))
                if not np.isfinite(L) or L <= 0:
                    L = float(np.nanmax(np.abs(diffs_concat))) if diffs_concat.size else 1.0
            else:
                L = 1.0
            log(f"[DIFF] Computed symmetric limit from percentiles: L = {L:.3f}")

        # Now render the diff maps (per height, per dataset)
        # 1) Dataset diffs
        for prefix, files in prefix_to_files.items():
            for z, fp in files:
                df_ds_full = read_height_file(fp, max_points=None)
                ds_h = build_means_df(df_ds_full, z, args.coord_precision)
                pr_h = pred_by_h.get(z, pd.DataFrame())
                m = merge_for_diff(pr_h, ds_h)
                if m.empty:
                    log(f"[diff:{prefix}] No overlap with predictions at {z} m; skipping.")
                    continue
                diffs = (m["pred"] - m["ds"]).to_numpy(dtype="float32")
                df_plot = m[["lat","lon"]].copy()
                out_fp = args.out_dir / f"map_diff_pred_minus_{prefix}_{z:03d}m.{args.format}"
                plot_diff_height_us_only(
                    df_plot, diffs, out_fp,
                    L=L,
                    s=args.point_size, alpha=args.alpha, dpi=args.dpi,
                    extent=tuple(args.extent), grid_dx=args.grid_dx, grid_dy=args.grid_dy,
                    ocean_face=args.ocean_color,
                    us_shapefile=args.us_shapefile,
                    verbose_counts=True,
                )
                log(f"[diff:{prefix}] Wrote {out_fp.name}")

        # 2) GWA diffs
        if df_gwa is not None and not df_gwa.empty:
            for z in gwa_heights:
                gwa_h = build_gwa_df(df_gwa, z, args.coord_precision)
                pr_h = pred_by_h.get(z, pd.DataFrame())
                m = merge_for_diff(pr_h, gwa_h)
                if m.empty:
                    log(f"[diff:gwa] No overlap with predictions at {z} m; skipping.")
                    continue
                diffs = (m["pred"] - m["ds"]).to_numpy(dtype="float32")
                df_plot = m[["lat","lon"]].copy()
                out_fp = args.out_dir / f"map_diff_pred_minus_gwa_{z:03d}m.{args.format}"
                plot_diff_height_us_only(
                    df_plot, diffs, out_fp,
                    L=L,
                    s=args.point_size, alpha=args.alpha, dpi=args.dpi,
                    extent=tuple(args.extent), grid_dx=args.grid_dx, grid_dy=args.grid_dy,
                    ocean_face=args.ocean_color,
                    us_shapefile=args.us_shapefile,
                    verbose_counts=True,
                )
                log(f"[diff:gwa] Wrote {out_fp.name}")

    log("All maps complete.")

if __name__ == "__main__":
    main()
