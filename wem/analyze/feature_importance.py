"""Visualize XGBoost feature importance (gain/cover/weight).

Expected input CSV columns (case-insensitive):
  feature, weight, gain, cover, total_gain, total_cover
Optionally: rank_gain (ignored if present)

Outputs (in --out-dir):
  - fi_bar_avg_gain.png / .pdf       : Horizontal bars sorted by average gain
  - fi_bar_total_gain.png / .pdf     : Horizontal bars sorted by total gain
  - fi_bubble_gain_vs_cover.png / .pdf : Bubble plot (avg gain vs avg cover;
                                         size=weight)
  - fi_normalized_summary.csv        : Percent-share table for quick copy into
                                       papers

Usage:
  wem-viz-fi --in feature_importance.csv --out-dir figs --top-n 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wem.constants import FEATURE_DISPLAY_MAP
from wem.utils.logging import log


def nice_name(s: str) -> str:
    """Map a raw feature name to a human-friendly display name."""
    s0 = str(s)
    return FEATURE_DISPLAY_MAP.get(s0, s0)


def read_fi_table(path: Path) -> pd.DataFrame:
    """Read and normalize a feature importance CSV.

    Parameters
    ----------
    path : Path
        CSV with columns feature, weight, gain, cover
        (and optionally total_gain, total_cover).

    Returns
    -------
    pd.DataFrame
        Normalized table with feature_disp, total_gain, total_cover.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    required = ["feature", "weight", "gain", "cover"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing required column '{r}' in {path}")
    out = pd.DataFrame(
        {
            "feature": df[cols["feature"]],
            "weight": pd.to_numeric(df[cols["weight"]], errors="coerce"),
            "gain": pd.to_numeric(df[cols["gain"]], errors="coerce"),
            "cover": pd.to_numeric(df[cols["cover"]], errors="coerce"),
        }
    )
    out["total_gain"] = (
        pd.to_numeric(df[cols["total_gain"]], errors="coerce")
        if "total_gain" in cols
        else out["gain"] * out["weight"]
    )
    out["total_cover"] = (
        pd.to_numeric(df[cols["total_cover"]], errors="coerce")
        if "total_cover" in cols
        else out["cover"] * out["weight"]
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["feature", "gain", "cover", "weight"]
    )
    out["feature_disp"] = out["feature"].map(nice_name)
    return out


def _fmt_thousands(x: float) -> str:
    return f"{x:,.0f}"


def barh_plot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Horizontal bar chart sorted by *metric*."""
    d = df.sort_values(metric, ascending=True).copy()
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.5))
    bars = ax.barh(y, d[metric].values, color="#4E79A7", alpha=0.9)
    ax.set_yticks(y, labels=d["feature_disp"].values)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for rect, v in zip(bars, d[metric].values):
        ax.text(
            rect.get_width(),
            rect.get_y() + rect.get_height() / 2,
            f" {v:,.2f}" if abs(v) >= 1e-3 else f" {v:.3g}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )
    ax.grid(axis="x", linestyle="--", color="#B8C2CC", alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def bubble_plot(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    """Bubble plot: avg gain vs avg cover, size = weight."""
    w = df["weight"].to_numpy(float)
    if np.nanmax(w) > 0:
        size = 200.0 * (w / np.nanmax(w)) + 30.0
    else:
        size = np.full_like(w, 60.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df["cover"],
        df["gain"],
        s=size,
        c="#59A14F",
        alpha=0.75,
        edgecolors="#2F4F2F",
        linewidths=0.6,
    )
    for _, r in df.iterrows():
        ax.text(
            r["cover"] * 1.005,
            r["gain"] * 1.005,
            r["feature_disp"],
            fontsize=9,
            alpha=0.9,
        )
    ax.set_xlabel("Average cover per split (samples)")
    ax.set_ylabel("Average gain per split")
    ax.set_title(
        "Feature splits: strength vs. breadth (size = split count / weight)"
    )
    ax.grid(True, linestyle="--", color="#B8C2CC", alpha=0.6)
    if np.nanmax(w) > 0:
        legend_sizes = [0.25, 0.5, 1.0]
        handles: List = []
        for frac in legend_sizes:
            handles.append(
                plt.scatter(
                    [],
                    [],
                    s=200 * frac + 30,
                    color="#59A14F",
                    alpha=0.75,
                    edgecolors="#2F4F2F",
                    linewidths=0.6,
                )
            )
        labels = [
            f"weight \u2248 {_fmt_thousands(frac * np.nanmax(w))}"
            for frac in legend_sizes
        ]
        ax.legend(
            handles,
            labels,
            title="Split count (weight)",
            loc="best",
            frameon=True,
        )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_normalized_summary(df: pd.DataFrame, out_csv: Path) -> None:
    """Write a percent-share table to CSV."""
    s = df.copy()
    for col in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        tot = s[col].sum()
        s[col + "_share"] = s[col] / tot if tot and np.isfinite(tot) else np.nan
    for col in ["gain", "total_gain", "weight", "cover"]:
        s[col + "_rank"] = (
            s[col].rank(ascending=False, method="dense").astype(int)
        )
    keep = [
        "feature",
        "feature_disp",
        "gain",
        "total_gain",
        "weight",
        "cover",
        "gain_share",
        "total_gain_share",
        "weight_share",
        "cover_share",
        "gain_rank",
        "total_gain_rank",
        "weight_rank",
        "cover_rank",
    ]
    s[keep].sort_values("gain_rank").to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Plot simple, clear visuals for XGBoost feature importance."
    )
    ap.add_argument(
        "--in",
        dest="infile",
        type=Path,
        required=True,
        help="CSV with feature importance columns.",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="Directory for plots."
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Plot top-N features (by the chosen metric).",
    )
    ap.add_argument("--dpi", type=int, default=300, help="Savefig DPI.")
    args = ap.parse_args()

    # Apply styling after arg parse
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": args.dpi,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_fi_table(args.infile)

    top_avg_gain = (
        df.sort_values("gain", ascending=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    top_total_gain = (
        df.sort_values("total_gain", ascending=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    bubble_df = pd.concat(
        [top_avg_gain, top_total_gain], ignore_index=True
    ).drop_duplicates("feature")

    barh_plot(
        top_avg_gain,
        metric="gain",
        title="Feature importance \u2014 Average gain per split",
        xlabel="Average gain (higher = stronger split improvements)",
        out_png=args.out_dir / "fi_bar_avg_gain.png",
        out_pdf=args.out_dir / "fi_bar_avg_gain.pdf",
    )

    barh_plot(
        top_total_gain,
        metric="total_gain",
        title="Feature importance \u2014 Total gain (overall contribution)",
        xlabel="Total gain (sum over all splits)",
        out_png=args.out_dir / "fi_bar_total_gain.png",
        out_pdf=args.out_dir / "fi_bar_total_gain.pdf",
    )

    bubble_plot(
        bubble_df,
        out_png=args.out_dir / "fi_bubble_gain_vs_cover.png",
        out_pdf=args.out_dir / "fi_bubble_gain_vs_cover.pdf",
    )

    write_normalized_summary(df, args.out_dir / "fi_normalized_summary.csv")

    log(f"Wrote figures to: {args.out_dir.resolve()}")
    log(
        f"Summary CSV: "
        f"{(args.out_dir / 'fi_normalized_summary.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
