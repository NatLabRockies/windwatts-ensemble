"""Machine-learning helper utilities for XGBoost training and LOOCV.

Includes feature engineering, neighbor-map construction, cohort balancing,
deterministic fold seeding, column presence filtering, and GWA helpers.
"""

from __future__ import annotations

__all__ = [
    "pick_present",
    "make_features",
    "build_neighbor_map",
    "balance_indices",
    "fold_seed",
    "_interp_gwa_row",
    "merge_gwa_feature",
]

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from wem.utils.logging import log
from wem.utils.power_law import fit_power_law_alpha
from wem.utils.sites import normalize_obs_type


# ---------------------------------------------------------------------------
# Column presence filter
# ---------------------------------------------------------------------------


def pick_present(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """Return only those column names from *candidates* that exist in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns are checked.
    candidates : list[str]
        Ordered list of candidate column names.

    Returns
    -------
    list[str]
        Subset of *candidates* present in ``df.columns``, in original order.
    """
    return [c for c in candidates if c in df.columns]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns to the training DataFrame.

    Adds:
    - ``is_gs``: binary flag (int8) indicating Gold Standard observation type.
    - ``aspect_sin``, ``aspect_cos``: sine/cosine encoding of ``aspect_deg``
      (only when the column is present).

    Parameters
    ----------
    df : pd.DataFrame
        Training data with at least an ``observation_type`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional feature columns.
    """
    out = df.copy()
    # binary flag for GS (not used as feature by default, but kept if requested)
    out["is_gs"] = (
        out["observation_type"]
        .astype(str)
        .map(normalize_obs_type)
        .eq("GS")
        .astype(np.int8)
    )
    # aspect -> sin/cos
    if "aspect_deg" in out.columns:
        rad = np.deg2rad(pd.to_numeric(out["aspect_deg"], errors="coerce"))
        out["aspect_sin"] = np.sin(rad)
        out["aspect_cos"] = np.cos(rad)
    return out


# ---------------------------------------------------------------------------
# Neighbor map for LOOCV exclusion
# ---------------------------------------------------------------------------


def build_neighbor_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build a mapping from GS station_id to its 10-km neighbor station_ids.

    Reads the pre-computed ``neighbors_10km_site_ids`` column (a
    comma-separated string of station ids).

    Parameters
    ----------
    df : pd.DataFrame
        Training data with ``station_id``, ``observation_type``, and
        ``neighbors_10km_site_ids`` columns.

    Returns
    -------
    dict[str, set[str]]
        Mapping from each Gold Standard station id to the set of
        neighboring station ids.
    """
    nbr_map: Dict[str, Set[str]] = {}
    if "neighbors_10km_site_ids" not in df.columns:
        return nbr_map
    is_gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS")
    uniq = df.loc[is_gs, ["station_id", "neighbors_10km_site_ids"]].drop_duplicates("station_id")
    for sid, s in zip(uniq["station_id"].astype(str), uniq["neighbors_10km_site_ids"].fillna("")):
        s = str(s).strip()
        nbr_map[sid] = set([] if s == "" else [x for x in s.split(",") if x != ""])
    return nbr_map


# ---------------------------------------------------------------------------
# Cohort balancing
# ---------------------------------------------------------------------------


def balance_indices(
    idx_asos: np.ndarray,
    idx_gs: np.ndarray,
    rng: np.random.Generator,
    strategy: str = "downsample",
) -> np.ndarray:
    """Balance GS and ASOS training indices so both contribute equally.

    Parameters
    ----------
    idx_asos : np.ndarray
        Row indices belonging to the ASOS cohort.
    idx_gs : np.ndarray
        Row indices belonging to the Gold Standard cohort.
    rng : np.random.Generator
        Random number generator for reproducible sampling.
    strategy : str
        ``"downsample"`` (default) reduces the majority to the minority
        size; ``"upsample"`` inflates the minority (with replacement) to
        match the majority.

    Returns
    -------
    np.ndarray
        Concatenated, balanced training indices.
    """
    n_asos = idx_asos.size
    n_gs = idx_gs.size
    if n_asos == 0 or n_gs == 0:
        return idx_asos if n_gs == 0 else idx_gs
    if strategy == "upsample":
        n_target = max(n_asos, n_gs)
        if n_asos < n_target:
            add = rng.choice(idx_asos, size=n_target - n_asos, replace=True)
            idx_asos = np.concatenate([idx_asos, add])
        elif n_gs < n_target:
            add = rng.choice(idx_gs, size=n_target - n_gs, replace=True)
            idx_gs = np.concatenate([idx_gs, add])
    else:
        # downsample
        n_target = min(n_asos, n_gs)
        if n_asos > n_target:
            idx_asos = rng.choice(idx_asos, size=n_target, replace=False)
        if n_gs > n_target:
            idx_gs = rng.choice(idx_gs, size=n_target, replace=False)
    return np.concatenate([idx_asos, idx_gs])


# ---------------------------------------------------------------------------
# Deterministic per-fold seed
# ---------------------------------------------------------------------------


def fold_seed(base_seed: int, sid: str) -> int:
    """Compute a deterministic per-station seed for LOOCV reproducibility.

    Parameters
    ----------
    base_seed : int
        Global random seed.
    sid : str
        Station identifier.

    Returns
    -------
    int
        A 32-bit integer seed derived from *base_seed* and *sid*.
    """
    h = hashlib.sha1((sid + str(base_seed)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit int


# ---------------------------------------------------------------------------
# GWA (Global Wind Atlas) helpers
# ---------------------------------------------------------------------------


def _interp_gwa_row(row: pd.Series, target_h: float, gwa_col: str) -> Optional[float]:
    """Return gwa_interp for this row: prefer provided column, else fit from available gwa_* columns."""
    # Use precomputed column if present
    if gwa_col in row and pd.notna(row[gwa_col]):
        try:
            val = float(row[gwa_col])
            return val if np.isfinite(val) and val > 0 else None
        except Exception:
            pass
    # Else fit from any subset of gwa_10/50/100/150 that exist
    cand_cols = [(10.0, "gwa_10"), (50.0, "gwa_50"), (100.0, "gwa_100"), (150.0, "gwa_150")]
    z_list, u_list = [], []
    for z, c in cand_cols:
        if c in row and pd.notna(row[c]):
            try:
                u = float(row[c])
                if np.isfinite(u) and u > 0:
                    z_list.append(z); u_list.append(u)
            except Exception:
                continue
    if len(z_list) < 2:
        return None
    fit = fit_power_law_alpha(np.array(z_list), np.array(u_list))
    if fit is None:
        return None
    A, alpha = fit
    if not (np.isfinite(A) and np.isfinite(alpha)) or target_h <= 0:
        return None
    return float(A * (float(target_h) ** alpha))


def merge_gwa_feature(
    df: pd.DataFrame,
    gwa_path: Optional[Path],
    gwa_col: str = "gwa_interp",
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Merge per-site/height GWA mean into df on (station_id,height_m).
    Returns (df_with_gwa, gwa_feature_name or None).
    """
    if gwa_path is None:
        return df, None
    if not gwa_path.exists():
        raise FileNotFoundError(f"GWA file not found: {gwa_path}")

    log(f"[INFO] Loading GWA table: {gwa_path}")
    if gwa_path.suffix.lower() in (".parquet", ".pq"):
        gwa = pd.read_parquet(gwa_path)
    else:
        gwa = pd.read_csv(gwa_path, low_memory=False)

    # Ensure keys exist & are typed correctly
    for c in ["station_id", "height_m"]:
        if c not in gwa.columns:
            raise ValueError(f"GWA file must contain '{c}'")
    gwa["station_id"] = gwa["station_id"].astype(str)
    gwa["height_m"] = pd.to_numeric(gwa["height_m"], errors="coerce")

    # Build/ensure gwa_interp column
    if gwa_col not in gwa.columns:
        log(f"[INFO] '{gwa_col}' not found; attempting power-law interpolation from available gwa_* columns.")
        # Vectorized apply with fallback
        gwa[gwa_col] = gwa.apply(lambda r: _interp_gwa_row(r, float(r["height_m"]), gwa_col), axis=1)

    # Keep only needed columns
    keep = ["station_id", "height_m", gwa_col]
    keep += [c for c in ["gwa_10", "gwa_50", "gwa_100", "gwa_150"] if c in gwa.columns]  # optional for debug
    gwa = gwa[keep].copy()

    # Merge into training df
    n_before = len(df)
    df = df.merge(gwa, on=["station_id", "height_m"], how="left")
    n_after = len(df)
    have = int(pd.Series(df[gwa_col]).notna().sum())
    log(f"[INFO] GWA merge done: rows {n_before}->{n_after}, with {have} non-null '{gwa_col}' values.")

    return df, gwa_col
