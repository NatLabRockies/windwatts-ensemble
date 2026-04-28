#!/usr/bin/env python3
"""
Make per-height maps from merged per-dataset quantile files (HRRR, WTK, WTK-LED),
from a single merged ML predictions file, and from GWA values precomputed at the
ML inference heights (via inference_table_w_gwa.*).

Custom color ramp (fixed 0->10+ m/s):
    0 m/s  -> white (#FFFFFF)
    ->      -> blue  (#5A8DC1)
    ->      -> yellow(#F7E873)
    ->      -> red   (#C33732)
    >=10 m/s-> purple(#8F3A63)   (colorbar extend='max' shows 10+)

Outputs (to --out-dir):
  - map_<prefix>_mean_<height>m.(png|pdf)  for datasets
  - map_pred_mean_<height>m.(png|pdf)      for predictions
  - map_gwa_mean_<height>m.(png|pdf)       for GWA (using gwa_interp)

Notes
-----
* GWA is read from an inference table that already contains the vertically
  interpolated GWA at the target hub heights (column: gwa_interp). We
  deduplicate to one row per (lat,lon,height_m) before mapping.
* If --global-scale is requested, percentiles are computed across ALL sources
  (datasets, predictions, and GWA) and applied to every map.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


import cartopy.crs as ccrs
import cartopy.feature as cfeature

from wem.utils.logging import log
from wem.utils.quantiles import mean_from_quantiles
from wem.utils.plotting import make_custom_cmap, wrap_lon180, mask_points_to_us

QCOLS = [f"q{p:03d}" for p in range(101)]

CUSTOM_CMAP = make_custom_cmap()

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
    usecols = ["lat", "lon"] + QCOLS
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
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
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

    # Deduplicate to one row per (lat,lon,height_m)
    df = df.sort_values(["lat", "lon", "height_m"]).drop_duplicates(["lat", "lon", "height_m"])
    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    df = df.reset_index(drop=True)
    return df

def unique_gwa_heights(df_gwa: pd.DataFrame) -> List[int]:
    hs = pd.to_numeric(df_gwa["height_m"], errors="coerce").dropna().unique()
    return sorted(int(round(float(h))) for h in hs)

# -------- Robust U.S. mask helpers (kept inline: build_us_prepared variants) --------
# Note: _build_us_prepared_from_cartopy_cache, _build_us_prepared_from_path,
# and _build_us_prepared are specific to this script's --us-shapefile fallback
# pattern and are kept inline. The core mask_points_to_us is imported from
# wem.utils.plotting.

def plot_one_height_us_only(
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

    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

    cmap = CUSTOM_CMAP
    bottom_color = cmap(norm(vmin))  # color at vmin
    top_color = cmap(norm(vmax))  # color at vmin
    cmap.set_under(bottom_color)     # values < vmin (and lower triangle) use this
    cmap.set_over(top_color)     # values < vmin (and lower triangle) use this


    mappable = ax.scatter(
        lonp, latp,
        c=vp,
        s=s * 1.4,
        marker="s",
        cmap=cmap,
        norm=norm,          # <-- use norm instead of passing vmin/vmax here
        transform=pc,
        linewidths=0,
        alpha=alpha,
        zorder=4,
    )

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=ocean_face, edgecolor="none", zorder=5)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=ocean_face, edgecolor="none", zorder=6)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, zorder=7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=7)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, zorder=7)

    import matplotlib.ticker as mticker
    gl = ax.gridlines(
        draw_labels=False,  # no lat/lon labels
        linewidth=0.4,
        color="0.6",
        alpha=0.6,
        linestyle="--",
    )
    gl.xlocator = mticker.MultipleLocator(base=grid_dx)
    gl.ylocator = mticker.MultipleLocator(base=grid_dy)

    # remove axis ticks entirely
    ax.set_xticks([])
    ax.set_yticks([])

    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])

    cb = fig.colorbar(mappable, cax=cax, extend="both")  # no `extend` -> rectangular bar
    cb.set_label(r"Mean wind speed ($m\,s^{-1}$)", fontsize=16)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontsize(14)

    try:
        cb.set_ticks(np.linspace(vmin, vmax, int(vmax - vmin) + 1))
    except Exception:
        pass

    # Add height label in bottom-right corner inside the axes


    fig.savefig(out_fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- Global scale helpers ----------
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

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Create per-height mean wind maps (datasets, ML, and GWA) with 0->10+ color ramp.")
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory with <prefix>_quantiles_*m.(csv|parquet|pq)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write map images")
    ap.add_argument("--prefix", type=str, default="hrrr,wtk,wtk_led,era5",
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
                    help="Compute ONE vmin/vmax across ALL sources (datasets, predictions, and GWA) from percentiles.")
    ap.add_argument("--scale-pct", type=str, default="1,99", help="Percentiles for global scale (if --global-scale)")
    ap.add_argument("--vmin", type=float, default=0.0, help="Color scale minimum (default 0 for white)")
    ap.add_argument("--vmax", type=float, default=10.0, help="Color scale maximum (default 10 for purple/10+)")
    ap.add_argument("--extent", type=float, nargs=4, default=[-125, -66, 24, 50],
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    help="Map extent in degrees (Plate Carree). Default: CONUS.")
    ap.add_argument("--grid-dx", type=float, default=5.0, help="Longitude grid spacing (deg)")
    ap.add_argument("--grid-dy", type=float, default=5.0, help="Latitude grid spacing (deg)")
    ap.add_argument("--ocean-color", type=str, default="white", help="Fill color for oceans/lakes")
    ap.add_argument("--us-shapefile", type=Path, default=None,
                    help="Path to a local U.S. polygon file (e.g., Natural Earth admin_0_countries.shp or a U.S.-only GeoJSON).")
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

    # ----- Parse percentile bounds -----
    try:
        lo_pct, hi_pct = (float(x) for x in args.scale_pct.split(","))
    except Exception:
        lo_pct, hi_pct = 1.0, 99.0

    # ----- Compute ONE global color scale if requested -----
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
        log(f"[GLOBAL] Color scale from percentiles {lo_pct},{hi_pct}: vmin={vmin_g:.3f}, vmax={vmax_g:.3f}")
    else:
        vmin_g = float(args.vmin)
        vmax_g = float(args.vmax)

    # ----- Plot dataset files -----
    for prefix, files in prefix_to_files.items():
        for z, fp in files:
            log(f"[{prefix}] Reading {fp.name} ...")
            df = read_height_file(fp, max_points=args.max_points)
            values = mean_from_quantiles(df)

            # Decide vmin/vmax
            vmin, vmax = (vmin_g, vmax_g) if args.global_scale else (float(args.vmin), float(args.vmax))

            out_fp = args.out_dir / f"map_{prefix}_mean_{z:03d}m.{args.format}"
            plot_one_height_us_only(
                df, values, out_fp,
                vmin=vmin, vmax=vmax,
                s=args.point_size, alpha=args.alpha, dpi=args.dpi,
                extent=tuple(args.extent), grid_dx=args.grid_dx, grid_dy=args.grid_dy,
                ocean_face=args.ocean_color,
                us_shapefile=args.us_shapefile,
                verbose_counts=True,
            )
            log(f"[{prefix}] Wrote {out_fp.name}")

    # ----- Plot predictions per-height -----
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
            plot_one_height_us_only(
                sub, values, out_fp,
                vmin=vmin, vmax=vmax,
                s=args.point_size, alpha=args.alpha, dpi=args.dpi,
                extent=tuple(args.extent), grid_dx=args.grid_dx, grid_dy=args.grid_dy,
                ocean_face=args.ocean_color,
                us_shapefile=args.us_shapefile,
                verbose_counts=True,
            )
            log(f"[pred] Wrote {out_fp.name}")

    # ----- Plot GWA per-height (using gwa_interp) -----
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
            # Ensure the DataFrame has lat/lon for plotting
            sub_plot = sub[["lat", "lon"]].copy()
            plot_one_height_us_only(
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

    log("All maps complete.")

if __name__ == "__main__":
    main()
