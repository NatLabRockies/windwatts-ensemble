"""ASOS outlier detection, visualization, and filtering.

Core ideas:
  * Compute site-level metrics (bias vs chosen reference OR RMSE vs consensus).
  * Robust z-score (MAD) to flag outliers (default |z| > 3.5).
  * Visualize histogram, optional map, and top-K station panels.
  * Filter ONLY ASOS rows; GS rows are always retained.

Dependencies: pandas, numpy, matplotlib. Cartopy is optional (map skipped
if missing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from wem.utils.logging import log
from wem.utils.ml import pick_present
from wem.utils.sites import normalize_obs_type


# ───────────────────────── helpers ─────────────────────────


def robust_zscores(x: np.ndarray) -> np.ndarray:
    """MAD-based robust z-scores (return NaN for all-NaN or zero spread).

    Parameters
    ----------
    x : np.ndarray
        1-D array of values.

    Returns
    -------
    np.ndarray
        Robust z-scores, same shape as *x*.
    """
    x = np.asarray(x, dtype="float64")
    m = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - m))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = np.nanstd(x)
    if not np.isfinite(s) or s <= 1e-12:
        return np.full_like(x, np.nan, dtype=float)
    return (x - m) / s


def build_reference_columns(
    df: pd.DataFrame, ref: str
) -> Tuple[str, List[str]]:
    """Return (ref_col_name, used_cols_for_consensus)."""
    ref = ref.lower()
    if ref in {"era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate"}:
        if ref not in df.columns:
            raise ValueError(
                f"Reference column '{ref}' not found in data."
            )
        return ref, [ref]
    if ref == "consensus":
        model_cols = pick_present(
            df,
            ["era5", "hrrr", "wtk", "wtk_led_conus", "wtk_led_climate"],
        )
        if not model_cols:
            raise ValueError(
                "No model columns available to build a consensus reference."
            )
        return "consensus_ref", model_cols
    raise ValueError(f"Unknown reference '{ref}'")


def compute_station_metrics(
    df: pd.DataFrame,
    ref_choice: str,
    monotonic_tol: float,
    use_elev_diff: bool,
) -> pd.DataFrame:
    """Compute per-ASOS-station QC metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Long table with observation_type, station_id, observation, etc.
    ref_choice : str
        Reference dataset or ``"consensus"``.
    monotonic_tol : float
        Allowed drop between adjacent quantiles before counting a violation.
    use_elev_diff : bool
        Whether to compute elevation mismatch metric.

    Returns
    -------
    pd.DataFrame
        One row per station with metrics, z-scores, and flags.
    """
    asos = df[
        df["observation_type"].astype(str).map(normalize_obs_type).eq("ASOS")
    ].copy()
    if asos.empty:
        raise ValueError("No ASOS rows found (observation_type=='ASOS').")

    ref_col, consensus_cols = build_reference_columns(asos, ref_choice)
    if ref_col == "consensus_ref":
        asos[ref_col] = np.nanmean(
            asos[consensus_cols].to_numpy(dtype="float64"), axis=1
        )

    # Pre-convert observation column to numeric once
    asos["_obs_numeric"] = pd.to_numeric(
        asos["observation"], errors="coerce"
    )

    grp = asos.groupby("station_id", sort=False)
    n_q = grp["_obs_numeric"].apply(lambda s: np.isfinite(s).sum())
    obs_max = grp["_obs_numeric"].apply(lambda s: np.nanmax(s.to_numpy()))
    obs_q95 = grp["_obs_numeric"].apply(
        lambda s: np.nanpercentile(s.to_numpy(), 95)
    )
    obs_med = grp["_obs_numeric"].apply(
        lambda s: np.nanmedian(s.to_numpy())
    )

    def count_monotonic_violations(site_df: pd.DataFrame) -> int:
        s = site_df.sort_values("qnum")
        v = s["_obs_numeric"].to_numpy(dtype="float64")
        good = np.isfinite(v)
        v = v[good]
        if v.size < 2:
            return 0
        diffs = np.diff(v)
        return int(np.sum(diffs < -abs(monotonic_tol)))

    mono = grp.apply(count_monotonic_violations)

    if ref_choice == "era5":
        bias_med = grp.apply(
            lambda g: float(
                np.nanmedian(
                    g["_obs_numeric"].to_numpy(dtype="float64")
                    - pd.to_numeric(g["era5"], errors="coerce").to_numpy(
                        dtype="float64"
                    )
                )
            )
        )
        metric = bias_med.rename("metric")
        metric_label = "bias_med_vs_ERA5"
    elif ref_choice == "consensus":

        def rmse_vs_consensus(g: pd.DataFrame) -> float:
            o = g["_obs_numeric"].to_numpy(dtype="float64")
            c = np.nanmean(
                g[
                    pick_present(
                        g,
                        [
                            "era5",
                            "hrrr",
                            "wtk",
                            "wtk_led_conus",
                            "wtk_led_climate",
                        ],
                    )
                ].to_numpy(dtype="float64"),
                axis=1,
            )
            good = np.isfinite(o) & np.isfinite(c)
            if not np.any(good):
                return np.nan
            return float(np.sqrt(np.mean((o[good] - c[good]) ** 2)))

        metric = grp.apply(rmse_vs_consensus).rename("metric")
        metric_label = "rmse_vs_consensus"
    else:

        def bias_med_generic(g: pd.DataFrame) -> float:
            o = g["_obs_numeric"].to_numpy(dtype="float64")
            r = pd.to_numeric(g[ref_col], errors="coerce").to_numpy(
                dtype="float64"
            )
            good = np.isfinite(o) & np.isfinite(r)
            if not np.any(good):
                return np.nan
            return float(np.nanmedian(o[good] - r[good]))

        metric = grp.apply(bias_med_generic).rename("metric")
        metric_label = f"bias_med_vs_{ref_choice.upper()}"

    lat = grp["lat"].first()
    lon = grp["lon"].first()

    out = pd.DataFrame(
        {
            "station_id": n_q.index.astype(str),
            "lat": lat.values.astype(float),
            "lon": lon.values.astype(float),
            "n_q": n_q.values.astype(int),
            "obs_max": obs_max.values.astype(float),
            "obs_q95": obs_q95.values.astype(float),
            "obs_med": obs_med.values.astype(float),
            "monotonic_violations": mono.values.astype(int),
            "metric": metric.values.astype(float),
        }
    )
    out["metric_label"] = metric_label

    if use_elev_diff and (
        "elev_m" in asos.columns or "elevation_m" in asos.columns
    ):
        elev_src = "elev_m" if "elev_m" in asos.columns else None
        dem_src = "elevation_m" if "elevation_m" in asos.columns else None
        if elev_src and dem_src:
            elev_diff = grp.apply(
                lambda g: float(
                    np.nanmedian(
                        np.abs(
                            pd.to_numeric(
                                g[elev_src], errors="coerce"
                            ).to_numpy(dtype="float64")
                            - pd.to_numeric(
                                g[dem_src], errors="coerce"
                            ).to_numpy(dtype="float64")
                        )
                    )
                )
            )
            out["elev_diff"] = elev_diff.values.astype(float)

    out["zscore"] = robust_zscores(out["metric"].to_numpy(dtype="float64"))
    return out


def plot_histogram(
    sta: pd.DataFrame, outdir: Path, z_thresh: float
) -> None:
    """Plot metric histogram with flagged stations highlighted."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    m = sta["metric"].to_numpy(dtype="float64")
    good = np.isfinite(m)
    flagged = np.abs(sta["zscore"]) > z_thresh

    ax.hist(m[good & ~flagged], bins=40, alpha=0.8, label="Inliers")
    ax.hist(m[good & flagged], bins=100, alpha=0.8, label="Flagged")
    ax.set_title(f"Station metric histogram (|z|>{z_thresh})")
    ax.set_xlabel(
        sta["metric_label"].iloc[0]
        if "metric_label" in sta.columns
        else "metric"
    )
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "metric_hist.png", bbox_inches="tight")
    plt.close(fig)


def plot_map(
    sta: pd.DataFrame, outdir: Path, z_thresh: float
) -> None:
    """Plot a map of station metrics with flagged stations marked."""
    if not HAS_CARTOPY:
        log("[WARN] Cartopy not available; skipping map.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6), dpi=140)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efefe8")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#cbd5e1")
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"), edgecolor="#555555", linewidth=0.5
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor="#555555",
        linewidth=0.5,
    )
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="#cbd5e1",
        edgecolor="#7aa2cc",
        linewidth=0.4,
    )

    flagged = np.abs(sta["zscore"]) > z_thresh
    sc1 = ax.scatter(
        sta.loc[~flagged, "lon"],
        sta.loc[~flagged, "lat"],
        c=sta.loc[~flagged, "metric"],
        s=18,
        cmap="viridis",
        vmin=np.nanpercentile(sta["metric"], 5),
        vmax=np.nanpercentile(sta["metric"], 95),
        edgecolors="#333",
        linewidths=0.2,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.scatter(
        sta.loc[flagged, "lon"],
        sta.loc[flagged, "lat"],
        c=sta.loc[flagged, "metric"],
        s=34,
        cmap="coolwarm",
        edgecolors="#111",
        linewidths=0.4,
        alpha=0.95,
        marker="x",
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    ax.set_title("ASOS stations \u2014 metric (flagged marked with 'x')")
    cbar = plt.colorbar(sc1, ax=ax, pad=0.01, fraction=0.035)
    cbar.set_label(
        sta["metric_label"].iloc[0]
        if "metric_label" in sta.columns
        else "metric"
    )
    fig.tight_layout()
    fig.savefig(outdir / "metric_map.png", bbox_inches="tight")
    plt.close(fig)


def plot_station_panels(
    df: pd.DataFrame,
    sta: pd.DataFrame,
    outdir: Path,
    top_k: int = 20,
) -> None:
    """Plot quantile curves for the worst-z-score stations."""
    outdir.mkdir(parents=True, exist_ok=True)

    worst = sta.copy()
    worst["abs_z"] = np.abs(worst["zscore"])
    worst = worst.sort_values("abs_z", ascending=False).head(top_k)

    model_cols = [
        c
        for c in [
            "era5",
            "hrrr",
            "wtk",
            "wtk_led_conus",
            "wtk_led_climate",
        ]
        if c in df.columns
    ]

    for sid in worst["station_id"]:
        sub = df[
            (df["station_id"].astype(str) == str(sid))
            & (
                df["observation_type"]
                .astype(str)
                .map(normalize_obs_type)
                .eq("ASOS")
            )
        ].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("qnum")

        obs = pd.to_numeric(sub["observation"], errors="coerce").to_numpy(
            dtype="float64"
        )
        q = pd.to_numeric(sub["qnum"], errors="coerce").to_numpy(
            dtype="int64"
        )

        fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
        ax.plot(q, obs, lw=2.0, label="ASOS obs", color="#1f77b4")

        if "era5" in sub.columns:
            ax.plot(
                q,
                pd.to_numeric(sub["era5"], errors="coerce"),
                lw=1.3,
                alpha=0.9,
                label="ERA5",
                color="#ff7f0e",
            )
        if model_cols:
            cons = np.nanmean(
                sub[model_cols].to_numpy(dtype="float64"), axis=1
            )
            ax.plot(
                q,
                cons,
                lw=1.3,
                alpha=0.9,
                label="Consensus",
                color="#2ca02c",
            )

        ax.set_title(f"Station {sid} \u2014 quantile curves")
        ax.set_xlabel("Quantile number (0..100)")
        ax.set_ylabel("Wind speed (m/s)")
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"station_{sid}.png", bbox_inches="tight")
        plt.close(fig)


# ───────────────────────── main ─────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="ASOS outlier detection, visualization, and filtering."
    )
    ap.add_argument(
        "--infile",
        type=Path,
        default=Path(
            "combined_quantiles_long_with_topo_loocv.csv"
        ),
    )
    ap.add_argument(
        "--outfile",
        type=Path,
        default=Path(
            "combined_quantiles_long_with_topo_loocv_asos_filtered.csv"
        ),
    )
    ap.add_argument(
        "--summary_out",
        type=Path,
        default=Path("qc/asos_station_summary.csv"),
    )
    ap.add_argument(
        "--plots_dir", type=Path, default=Path("qc/plots")
    )
    ap.add_argument(
        "--ref",
        type=str,
        default="era5",
        choices=[
            "era5",
            "hrrr",
            "wtk",
            "wtk_led_conus",
            "wtk_led_climate",
            "consensus",
        ],
        help="Reference for the outlier metric.",
    )
    ap.add_argument(
        "--z_thresh",
        type=float,
        default=3.5,
        help="Robust z-score threshold on the chosen metric.",
    )
    ap.add_argument(
        "--min_qrows",
        type=int,
        default=50,
        help="Minimum #quantiles per ASOS station to consider.",
    )
    ap.add_argument(
        "--max_ws",
        type=float,
        default=60.0,
        help="Hard cap for plausible wind speed (m/s).",
    )
    ap.add_argument(
        "--monotonic_tol",
        type=float,
        default=0.0,
        help="Allowed drop between adjacent quantiles.",
    )
    ap.add_argument(
        "--elev_mismatch_thresh",
        type=float,
        default=None,
        help="If set, flag stations with |elev_m - DEM_elevation_m| "
        "above this (meters).",
    )
    ap.add_argument(
        "--top_k_plots",
        type=int,
        default=20,
        help="How many worst stations to plot panels for.",
    )
    ap.add_argument(
        "--no_map",
        action="store_true",
        help="Skip drawing the Cartopy map.",
    )
    args = ap.parse_args()

    if not args.infile.exists():
        raise FileNotFoundError(args.infile)
    log(f"[INFO] Loading: {args.infile}")
    df = pd.read_csv(
        args.infile, dtype={"station_id": str}, low_memory=False
    )

    use_elev = args.elev_mismatch_thresh is not None
    sta = compute_station_metrics(
        df=df,
        ref_choice=args.ref,
        monotonic_tol=args.monotonic_tol,
        use_elev_diff=use_elev,
    )

    sta["flag_cov"] = sta["n_q"] < int(args.min_qrows)
    sta["flag_wsmax"] = sta["obs_max"] > float(args.max_ws)
    sta["flag_mono"] = sta["monotonic_violations"] > 0
    sta["flag_metric"] = np.abs(sta["zscore"]) > float(args.z_thresh)

    if use_elev and "elev_diff" in sta.columns:
        sta["flag_elev"] = sta["elev_diff"] > float(
            args.elev_mismatch_thresh
        )
    else:
        sta["flag_elev"] = False

    sta["is_outlier"] = (
        sta["flag_metric"]
        | sta["flag_cov"]
        | sta["flag_wsmax"]
        | sta["flag_mono"]
        | sta["flag_elev"]
    )

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    sta.to_csv(args.summary_out, index=False)
    log(f"[INFO] Wrote station summary -> {args.summary_out}")

    plot_histogram(sta, args.plots_dir, z_thresh=args.z_thresh)
    if not args.no_map:
        plot_map(sta, args.plots_dir, z_thresh=args.z_thresh)
    plot_station_panels(
        df, sta, args.plots_dir, top_k=int(args.top_k_plots)
    )

    outlier_sids = set(
        sta.loc[sta["is_outlier"], "station_id"].astype(str)
    )
    log(
        f"[INFO] Flagged ASOS stations: {len(outlier_sids)} / {len(sta)}"
    )

    mask_asos = (
        df["observation_type"]
        .astype(str)
        .map(normalize_obs_type)
        .eq("ASOS")
        .to_numpy()
    )
    mask_outlier_stn = (
        df["station_id"].astype(str).isin(outlier_sids).to_numpy()
    )

    keep = ~(mask_asos & mask_outlier_stn)
    df_filtered = df.loc[keep].reset_index(drop=True)

    log(f"[INFO] Writing filtered table -> {args.outfile}")
    df_filtered.to_csv(args.outfile, index=False)
    log(
        f"[INFO] Done. Original rows: {len(df)} | Kept: "
        f"{len(df_filtered)} | Dropped: {len(df) - len(df_filtered)}"
    )
    n_asos_rows = int(mask_asos.sum())
    n_asos_rows_kept = int((mask_asos & keep).sum())
    log(
        f"[INFO] ASOS rows kept: {n_asos_rows_kept}/{n_asos_rows} "
        f"(dropped {n_asos_rows - n_asos_rows_kept})"
    )


if __name__ == "__main__":
    main()
