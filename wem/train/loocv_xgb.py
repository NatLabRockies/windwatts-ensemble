#!/usr/bin/env python3
"""
Parallel LOOCV-by-GS-station for XGB regression with per-fold GS=ASOS balancing,
now with optional Global Wind Atlas (GWA) mean-speed feature.

- Target: 'observation' (m/s)
- Evaluation: GS stations only, each fold excludes that station and its 10-km neighbors
- Training: ASOS + GS (minus exclusions), balanced GS:ASOS (downsample or upsample)
- Optional GWA feature: per-site/height mean wind speed (gwa_interp), repeated across quantiles.
  If your GWA table lacks 'gwa_interp', it will be computed from any of
  {gwa_10,gwa_50,gwa_100,gwa_150} via power-law fit ln(U)=ln(A)+alpha*ln(z).

Input  : combined_quantiles_long_with_topo_loocv.csv (or .parquet)
Optional: site_height_ws_avg_with_gwa.csv (or .parquet) with station_id,height_m,[gwa_interp|gwa_10|gwa_50|gwa_100|gwa_150]
Output : CSV/Parquet with 'pred_observation'
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

# xgboost
try:
    from xgboost import XGBRegressor
except Exception as e:
    raise SystemExit("This script requires xgboost. Install with `pip install xgboost`") from e

# joblib for process-based parallelism + memmapping of large arrays
from joblib import Parallel, delayed

from wem.constants import AUX_FEATURE_MAP, DEFAULT_XGB_PARAMS, WIND_FEATURE_MAP
from wem.utils.logging import log
from wem.utils.sites import normalize_obs_type
from wem.utils.ml import pick_present, make_features, build_neighbor_map, balance_indices, fold_seed, merge_gwa_feature


# ────────────────────────── worker ──────────────────────────
def run_one_fold(
    sid: str,
    X_full: np.ndarray,         # (N, F) float32  (may contain NaNs for optional features like GWA)
    y_full: np.ndarray,         # (N,)   float32
    station_ids: np.ndarray,    # (N,)   str
    is_gs: np.ndarray,          # (N,)   bool
    nbr_map: Dict[str, Set[str]],
    args_dict: dict,
    require_finite_mask: Optional[np.ndarray] = None,  # boolean mask over columns to enforce finiteness
) -> Tuple[str, np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
    """
    Train one fold for station `sid` and return:
      (sid, test_idx, pred, (rmse, mae)) — metrics None if not computed.
    """
    # Unpack args
    balance_strategy = args_dict["balance_strategy"]
    val_frac         = args_dict["val_frac"]
    seed             = args_dict["seed"]
    xgb_params       = args_dict["xgb_params"]

    # Test rows = this GS station
    test_idx = np.where(station_ids == sid)[0]
    if test_idx.size == 0:
        return sid, np.array([], dtype=int), np.array([], dtype=np.float32), None

    # Exclusion set: this sid + its neighbors
    excl: Set[str] = {sid}
    if sid in nbr_map:
        excl |= set(nbr_map[sid])

    # Base training mask
    base_train_mask = (~np.isin(station_ids, np.fromiter(excl, dtype=station_ids.dtype))) & np.isfinite(y_full)

    # Candidate train rows
    base_idx = np.where(base_train_mask)[0]
    if base_idx.size == 0:
        return sid, test_idx, np.full(test_idx.shape, np.nan, dtype=np.float32), None

    # Require finite for a subset of columns (all except optional ones like GWA)
    if require_finite_mask is None:
        finite_feat_mask = np.all(np.isfinite(X_full[base_idx]), axis=1)
    else:
        req_cols = require_finite_mask
        if req_cols.any():
            finite_feat_mask = np.all(np.isfinite(X_full[base_idx][:, req_cols]), axis=1)
        else:
            finite_feat_mask = np.ones(base_idx.shape[0], dtype=bool)

    good_train_idx = base_idx[finite_feat_mask]
    if good_train_idx.size < 20:
        return sid, test_idx, np.full(test_idx.shape, np.nan, dtype=np.float32), None

    # Split indices by class for balancing
    idx_asos = good_train_idx[~is_gs[good_train_idx]]
    idx_gs   = good_train_idx[ is_gs[good_train_idx]]

    rng = np.random.default_rng(fold_seed(seed, sid))
    train_idx_bal = balance_indices(idx_asos, idx_gs, rng, strategy=balance_strategy)
    if train_idx_bal.size < 20:
        return sid, test_idx, np.full(test_idx.shape, np.nan, dtype=np.float32), None

    rng.shuffle(train_idx_bal)

    X_all = X_full[train_idx_bal]
    y_all = y_full[train_idx_bal]

    # Early-stopping validation split
    eval_set = None
    if val_frac > 0 and X_all.shape[0] > 50:
        n = X_all.shape[0]
        val_n = int(max(1, np.floor(val_frac * n)))
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:val_n]
        tr_idx  = idx[val_idx.size:]
        Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
        Xva, yva = X_all[val_idx], y_all[val_idx]
        eval_set = [(Xva, yva)]
    else:
        Xtr, ytr = X_all, y_all

    # Define & fit model (xgboost handles NaNs internally)
    model = XGBRegressor(**xgb_params)
    if eval_set is not None:
        model.fit(Xtr, ytr, eval_set=eval_set, verbose=False)
    else:
        model.fit(Xtr, ytr)

    # Predict on test rows
    Xt = X_full[test_idx]
    pred = model.predict(Xt).astype("float32")

    # Metrics for this station
    yt = y_full[test_idx]
    good = np.isfinite(yt) & np.isfinite(pred)
    metrics = None
    if np.any(good):
        yt_good, pt_good = yt[good], pred[good]
        rmse = float(np.sqrt(np.mean((pt_good - yt_good) ** 2)))
        mae  = float(np.mean(np.abs(pt_good - yt_good)))
        metrics = (rmse, mae)

    return sid, test_idx, pred, metrics


# ────────────────────────── main ──────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Parallel balanced (GS = ASOS) XGB with LOOCV-by-GS-site (10-km exclusion) + optional GWA mean feature.")
    ap.add_argument("--infile",  type=Path, default=Path("combined_quantiles_long_with_topo_loocv_10km.csv"),
                    help="Input CSV (or Parquet) with features + neighbor lists.")
    ap.add_argument("--outfile", type=Path, default=Path("TEST.csv"),
                    help="Output CSV with 'pred_observation' column.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite output file if it already exists.")
    ap.add_argument("--balance_strategy", type=str, default="downsample", choices=["downsample", "upsample"],
                    help="How to equalize GS and ASOS in the training set.")
    ap.add_argument("--gs_only", action="store_true", help="Only include GoldStandard sites in training data.")

    # Hyperparameters
    ap.add_argument("--learning_rate", type=float, default=DEFAULT_XGB_PARAMS['learning_rate'])
    ap.add_argument("--min_child_weight", type=float, default=DEFAULT_XGB_PARAMS['min_child_weight'])
    ap.add_argument("--subsample", type=float, default=DEFAULT_XGB_PARAMS['subsample'])
    ap.add_argument("--colsample_bytree", type=float, default=DEFAULT_XGB_PARAMS['colsample_bytree'])
    ap.add_argument("--n_estimators", type=int, default=DEFAULT_XGB_PARAMS['n_estimators'])
    ap.add_argument("--max_depth", type=int, default=DEFAULT_XGB_PARAMS['max_depth'])
    ap.add_argument("--early_stopping_rounds", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.0)

    # Parallel controls
    ap.add_argument("--n_jobs_outer", type=int, default=12, help="Parallel folds (processes). -1 = all cores.")
    ap.add_argument("--n_jobs_model", type=int, default=1, help="Threads per XGBoost model.")
    ap.add_argument("--memmap_min_bytes", type=str, default="None", help="Joblib memmap threshold ('1M','10M','None').")

    ap.add_argument("--seed", type=int, default=42)

    # Feature groups
    ap.add_argument("--wind_features", type=str,
                    default="hrrr,wtk,wtk_led_conus",
                    help="Comma-separated wind resource columns to include "
                         "(subset of: era5,hrrr,wtk,wtk_led_conus,wtk_led_climate).")
    ap.add_argument("--aux_features", type=str, default="latlon,height,elevation",
                    help=("Aux groups to include: "
                          "latlon(lat,lon), height(height_m), elevation(elevation_m), "
                          "slope(slope_deg), aspect(aspect_sin,aspect_cos). "
                          "Use '' to include none."))

    # GWA integration
    ap.add_argument("--gwa-file", type=Path, default=None,
                    help="CSV/Parquet with per-site/height GWA mean(s). Must have station_id,height_m and either "
                         "gwa_interp or some of gwa_10,gwa_50,gwa_100,gwa_150.")
    ap.add_argument("--gwa-col", type=str, default="gwa_interp",
                    help="Column name in --gwa-file to use as the mean-speed feature (default: gwa_interp).")
    ap.add_argument("--include-gwa", action="store_true",
                    help="Include the merged GWA mean feature as an additional predictor.")

    args = ap.parse_args()

    # --- Load base table ---
    path = args.infile
    if not path.exists():
        raise FileNotFoundError(path)
    log(f"[INFO] Loading: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"station_id": str}, low_memory=False)

    if args.gs_only:
        if "observation_type" not in df.columns:
            raise ValueError("Missing 'observation_type' column.")
        mask_gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS")
        df = df.loc[mask_gs].copy()
        if df.empty:
            raise SystemExit("No Gold Standard rows found.")

    df.reset_index(drop=True, inplace=True)

    # Required columns sanity
    for c in ["station_id", "lat", "lon", "observation", "observation_type", "height_m", "qnum"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    # Prepare features & neighbors
    df = make_features(df)
    nbr_map = build_neighbor_map(df)

    # --- Optional: merge GWA mean feature ---
    gwa_feature_name: Optional[str] = None
    if args.gwa_file is not None:
        df, gwa_feature_name = merge_gwa_feature(df, args.gwa_file, gwa_col=args.gwa_col)

    # Parse requested wind-resource features
    valid_wind = set(WIND_FEATURE_MAP)
    sel_wind = [s.strip() for s in str(args.wind_features).split(",") if s.strip()]
    bad = [w for w in sel_wind if w not in valid_wind]
    if bad:
        raise ValueError(f"--wind_features contains invalid names: {bad} (valid: {sorted(valid_wind)})")

    # --- Build feature list from requested wind + aux groups ---
    base_always = ["qnum"]  # always include quantile index

    req_winds = [w.strip().lower() for w in args.wind_features.split(",") if w.strip() != ""]
    req_aux   = [g.strip().lower() for g in args.aux_features.split(",") if g.strip() != ""]

    wind_cols = [WIND_FEATURE_MAP[w] for w in req_winds if w in WIND_FEATURE_MAP]
    aux_cols  = []
    for g in req_aux:
        aux_cols.extend(AUX_FEATURE_MAP.get(g, []))

    feature_candidates = base_always + wind_cols + aux_cols

    # Add GWA mean feature if requested and present
    gwa_in_use = False
    if args.include_gwa:
        if gwa_feature_name is None:
            log("[WARN] --include-gwa was set but no GWA feature was merged; ignoring.")
        else:
            feature_candidates.append(gwa_feature_name)
            gwa_in_use = True

    # Keep only columns that actually exist
    feat_cols = [c for c in feature_candidates if c in df.columns]
    log(f"[INFO] Using {len(feat_cols)} features: {feat_cols}")

    # Arrays / masks
    X_full = df[feat_cols].to_numpy(dtype="float32")  # may include NaNs (for optional features)
    y_full = pd.to_numeric(df["observation"], errors="coerce").to_numpy(dtype="float32")
    is_gs  = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS").to_numpy()
    station_ids = df["station_id"].astype(str).to_numpy()

    # GS stations for LOOCV
    gs_stations = pd.Series(station_ids[is_gs]).unique().tolist()
    log(f"[INFO] GS stations for LOOCV: {len(gs_stations)}")

    # Output column init
    df["pred_observation"] = np.nan

    # Build monotone constraint vector: +1 on 'qnum', 0 otherwise
    mon = [0] * len(feat_cols)
    if "qnum" in feat_cols:
        mon[feat_cols.index("qnum")] = 1
    mon_str = "(" + ",".join(map(str, mon)) + ")"

    # XGB model parameters for workers
    xgb_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        objective="reg:absoluteerror",
        n_jobs=args.n_jobs_model,
        random_state=args.seed,
        tree_method="hist",
        early_stopping_rounds=args.early_stopping_rounds,
        monotone_constraints=mon_str,
        # missing=np.nan,  # default is np.nan; XGBoost handles it
    )

    args_dict = dict(
        balance_strategy=args.balance_strategy,
        val_frac=args.val_frac,
        seed=args.seed,
        xgb_params=xgb_params,
    )

    # Build "require finite" mask over feature columns:
    # require finite for ALL features EXCEPT optional ones (currently only GWA).
    require_finite_mask = np.ones(len(feat_cols), dtype=bool)
    if gwa_in_use and (gwa_feature_name in feat_cols):
        require_finite_mask[feat_cols.index(gwa_feature_name)] = False

    # joblib memmap threshold
    max_nbytes = None if str(args.memmap_min_bytes).lower() == "none" else args.memmap_min_bytes

    log(f"[INFO] Parallelizing {len(gs_stations)} folds with n_jobs_outer={args.n_jobs_outer}, "
        f"n_jobs_model={args.n_jobs_model}, memmap_min_bytes={args.memmap_min_bytes}, "
        f"GWA_in_use={gwa_in_use}")

    # Run folds in parallel
    results = Parallel(
        n_jobs=args.n_jobs_outer,
        verbose=10,
        prefer="processes",
        max_nbytes=max_nbytes,
    )(
        delayed(run_one_fold)(
            sid, X_full, y_full, station_ids, is_gs, nbr_map, args_dict, require_finite_mask=require_finite_mask
        )
        for sid in gs_stations
    )

    # Collect predictions + metrics
    all_preds_list, all_truth_list = [], []
    per_site_rows = []

    for sid, test_idx, pred, metrics in results:
        if test_idx.size > 0 and pred.size == test_idx.size:
            df.loc[test_idx, "pred_observation"] = pred
            yt = y_full[test_idx]
            good = np.isfinite(yt) & np.isfinite(pred)
            if np.any(good):
                all_truth_list.append(yt[good])
                all_preds_list.append(pred[good])
                if metrics is not None:
                    rmse, mae = metrics
                    per_site_rows.append((sid, rmse, mae))

    # Save predictions
    out_path = args.outfile
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output file exists: {out_path}  (use --overwrite to replace)")
    log(f"[INFO] Saving predictions → {out_path}")
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    # Overall metrics on GS test rows
    if all_preds_list:
        P = np.concatenate(all_preds_list)
        T = np.concatenate(all_truth_list)
        rmse = float(np.sqrt(np.mean((P - T) ** 2)))
        mae  = float(np.mean(np.abs(P - T)))
        log(f"[METRIC] Balanced GS LOOCV (parallel) — RMSE={rmse:.4f} m/s, MAE={mae:.4f} m/s (N={len(T)})")
        if per_site_rows:
            mdf = pd.DataFrame(per_site_rows, columns=["station_id", "rmse", "mae"])
            best5  = mdf.nsmallest(5, "rmse")
            worst5 = mdf.nlargest(5, "rmse")
            log("[METRIC] Best 5 GS stations by RMSE:")
            log(best5.to_string(index=False))
            log("[METRIC] Worst 5 GS stations by RMSE:")
            log(worst5.to_string(index=False))
    else:
        log("[WARN] No GS predictions were produced; check inputs/neighbor lists.")

    log("[INFO] Done.")


if __name__ == "__main__":
    main()
