#!/usr/bin/env python3
"""
Train the final XGBoost model on ALL data (balanced GS <-> ASOS via downsampling)
and save model + artifacts for publication-quality reporting.

Now supports optional Global Wind Atlas (GWA) mean feature:
  - Merge on (station_id, height_m)
  - Use 'gwa_interp' if present; otherwise fit power-law from any subset of
    {gwa_10,gwa_50,gwa_100,gwa_150} per row to compute a height-matched mean.
  - XGBoost handles NaNs: rows without GWA are kept; only non-GWA features are required finite.

Inputs
------
- CSV/Parquet with columns:
    station_id, lat, lon, observation, observation_type, qnum,
    (optionally) height_m, elevation_m, slope_deg, aspect_deg, era5, hrrr, wtk, wtk_led_conus, wtk_led_climate, ...

Core behavior
-------------
- Builds features from requested wind + aux groups (defaults mirror your best config)
- Optional inclusion of GWA mean-speed feature (--gwa-file + --include-gwa)
- Filters rows with finite target and (required) finite features for training
- Downsamples the majority class so GS and ASOS contribute equally
- Trains single XGBRegressor with your optimal hyperparams on ALL (balanced) rows
- Saves:
    * xgb_model.json  (native model)
    * xgb_model.joblib (sklearn wrapper; optional if joblib available)
    * feature_importance.csv  (weight, gain, cover, total_gain, total_cover)
    * in_sample_predictions.csv (preds on ALL rows + residuals where obs available)
    * metrics_training.json (overall & by obs type: RMSE, MAE, N)
    * metadata.json (features, params, class counts pre/post-balance, seeds, versions)
    * feature_names.json
    * (optional) shap_summary.csv   (--shap true; mean|median|p95(|contrib|) per feature)

Usage
-----
python train_final_xgb_all_data.py \\
  --infile combined_quantiles_long_with_topo_loocv_10km.csv \\
  --out-dir model_final_all \\
  --n_jobs_model 8 \\
  --shap false \\
  --gwa-file site_height_ws_avg_with_gwa.csv --gwa-col gwa_interp --include-gwa
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Optional deps
try:
    import joblib  # for saving sklearn wrapper
except Exception:
    joblib = None

try:
    import shap  # optional, for SHAP summaries
except Exception:
    shap = None

# xgboost
try:
    from xgboost import XGBRegressor
except Exception as e:
    raise SystemExit("This script requires xgboost. Install with `pip install xgboost`") from e

from wem.constants import AUX_FEATURE_MAP, DEFAULT_XGB_PARAMS, WIND_FEATURE_MAP
from wem.utils.logging import log
from wem.utils.sites import normalize_obs_type
from wem.utils.ml import pick_present, make_features, balance_indices, merge_gwa_feature

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(good):
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    yt = y_true[good]; yp = y_pred[good]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae  = float(np.mean(np.abs(yp - yt)))
    return {"rmse": rmse, "mae": mae, "n": int(yt.size)}

def version_info() -> Dict[str, str]:
    info = {}
    try:
        import xgboost as xgb
        info["xgboost"] = xgb.__version__
    except Exception: pass
    try:
        info["numpy"] = np.__version__
    except Exception: pass
    try:
        info["pandas"] = pd.__version__
    except Exception: pass
    try:
        info["shap"] = shap.__version__ if shap is not None else "not-installed"
    except Exception: pass
    try:
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except Exception: pass
    return info


# ────────────────────────── main ──────────────────────────
def main():
    # Defaults: feature sets and balance strategy
    opt = {
        "seed": 42,
        "wind_features": "hrrr,wtk,wtk_led_conus",
        "aux_features": "latlon,height,elevation",
        "balance_strategy": "downsample",
    }

    ap = argparse.ArgumentParser(description="Train final XGB model on ALL data with GS<->ASOS downsampling, save artifacts (with optional GWA feature).")
    ap.add_argument("--infile", type=Path, default=Path("combined_quantiles_long_with_topo_loocv_10km.csv"),
                    help="Input CSV/Parquet with features & target.")
    ap.add_argument("--out-dir", type=Path, default=Path("model_final_all"),
                    help="Output folder (created if missing).")
    ap.add_argument("--n_jobs_model", type=int, default=8, help="Threads for XGBoost.")
    ap.add_argument("--learning_rate", type=float, default=DEFAULT_XGB_PARAMS["learning_rate"])
    ap.add_argument("--max_depth", type=int, default=DEFAULT_XGB_PARAMS["max_depth"])
    ap.add_argument("--min_child_weight", type=float, default=DEFAULT_XGB_PARAMS["min_child_weight"])
    ap.add_argument("--subsample", type=float, default=DEFAULT_XGB_PARAMS["subsample"])
    ap.add_argument("--colsample_bytree", type=float, default=DEFAULT_XGB_PARAMS["colsample_bytree"])
    ap.add_argument("--n_estimators", type=int, default=DEFAULT_XGB_PARAMS["n_estimators"])
    ap.add_argument("--seed", type=int, default=opt["seed"])
    ap.add_argument("--wind_features", type=str, default=opt["wind_features"],
                    help="Subset of: era5,hrrr,wtk,wtk_led_conus,wtk_led_climate")
    ap.add_argument("--aux_features", type=str, default=opt["aux_features"],
                    help="Groups: latlon,height,elevation,slope,aspect  (comma-sep)")
    ap.add_argument("--balance_strategy", type=str, default=opt["balance_strategy"],
                    choices=["downsample"], help="Final model uses downsample (fixed).")
    ap.add_argument("--val_frac", type=float, default=0.0, help="Optional tiny holdout for early stopping (0 = none).")
    ap.add_argument("--early_stopping_rounds", type=int, default=0, help=">0 requires val_frac>0.")
    ap.add_argument("--shap", type=str, default="false", choices=["true","false"],
                    help="Compute SHAP summary on a sample (can be slow).")
    ap.add_argument("--shap_sample", type=int, default=200000,
                    help="Max rows for SHAP (only if --shap true).")

    # GWA options (same behavior as LOOCV script)
    ap.add_argument("--gwa-file", type=Path, default=None,
                    help="CSV/Parquet with per-site/height GWA mean(s); must have station_id,height_m and either "
                         "gwa_interp or some of gwa_10,gwa_50,gwa_100,gwa_150.")
    ap.add_argument("--gwa-col", type=str, default="gwa_interp",
                    help="Column in --gwa-file to use as mean-speed feature (default: gwa_interp).")
    ap.add_argument("--include-gwa", action="store_true",
                    help="Include the merged GWA mean feature as an additional predictor.")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    if not args.infile.exists():
        raise FileNotFoundError(args.infile)
    log(f"Loading data: {args.infile}")
    if args.infile.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.infile)
    else:
        df = pd.read_csv(args.infile, dtype={"station_id": str}, low_memory=False)
    df = make_features(df)

    # Optional GWA merge (exactly as in LOOCV)
    gwa_feature_name: Optional[str] = None
    if args.gwa_file is not None:
        df, gwa_feature_name = merge_gwa_feature(df, args.gwa_file, gwa_col=args.gwa_col)

    # Feature groups
    req_winds = [w.strip().lower() for w in args.wind_features.split(",") if w.strip()]
    req_aux   = [g.strip().lower() for g in args.aux_features.split(",") if g.strip()]
    wind_cols = [WIND_FEATURE_MAP[w] for w in req_winds if w in WIND_FEATURE_MAP]
    aux_cols: List[str] = []
    for g in req_aux:
        aux_cols.extend(AUX_FEATURE_MAP.get(g, []))

    base_always = ["qnum"]
    feature_candidates = base_always + wind_cols + aux_cols

    # Add GWA feature if requested & merged
    gwa_in_use = False
    if args.include_gwa:
        if gwa_feature_name is None:
            log("[WARN] --include-gwa set but no GWA feature available after merge; proceeding without it.")
        else:
            feature_candidates.append(gwa_feature_name)
            gwa_in_use = True

    feat_cols = pick_present(df, feature_candidates)
    if not feat_cols:
        raise SystemExit("No usable features after applying wind/aux (+GWA) selection.")
    log(f"Feature set ({len(feat_cols)}): {feat_cols}")

    # Target & masks
    y_all = pd.to_numeric(df["observation"], errors="coerce").to_numpy(dtype="float32")
    is_gs = df["observation_type"].astype(str).map(normalize_obs_type).eq("GS").to_numpy()
    is_asos = df["observation_type"].astype(str).map(normalize_obs_type).eq("ASOS").to_numpy()

    X_all_df = df[feat_cols].copy()
    X_all_np = X_all_df.to_numpy(dtype="float32")

    # Require finite for all features EXCEPT optional GWA (matches LOOCV behavior)
    require_finite = np.ones(len(feat_cols), dtype=bool)
    if gwa_in_use and (gwa_feature_name in feat_cols):
        require_finite[feat_cols.index(gwa_feature_name)] = False
    if require_finite.any():
        finite_feat = np.all(np.isfinite(X_all_np[:, require_finite]), axis=1)
    else:
        finite_feat = np.ones(len(X_all_df), dtype=bool)

    finite_y = np.isfinite(y_all)
    base_train_mask = finite_feat & finite_y
    train_idx = np.where(base_train_mask)[0]

    # Balance GS vs ASOS on trainable rows (downsampling)
    rng = np.random.default_rng(args.seed)
    idx_asos = train_idx[is_asos[train_idx]]
    idx_gs   = train_idx[is_gs[train_idx]]
    train_idx_bal = balance_indices(idx_asos, idx_gs, rng, strategy="downsample")
    rng.shuffle(train_idx_bal)

    # Report counts
    counts = {
        "N_total_rows": int(len(df)),
        "N_finite_y": int(finite_y.sum()),
        "N_finite_required_features": int(finite_feat.sum()),
        "N_train_candidates": int(train_idx.size),
        "N_train_gs": int(idx_gs.size),
        "N_train_asos": int(idx_asos.size),
        "N_train_balanced": int(train_idx_bal.size),
        "N_train_bal_gs": int(np.sum(is_gs[train_idx_bal])),
        "N_train_bal_asos": int(np.sum(is_asos[train_idx_bal])),
    }
    log(f"Counts: {counts}")

    # Optional (very small) validation split if requested
    eval_set = None
    Xtr_df = X_all_df.iloc[train_idx_bal]
    ytr = y_all[train_idx_bal]
    if args.val_frac > 0 and args.early_stopping_rounds > 0 and Xtr_df.shape[0] > 50:
        n = len(train_idx_bal)
        val_n = int(max(1, np.floor(args.val_frac * n)))
        perm = rng.permutation(n)
        val_sel = train_idx_bal[perm[:val_n]]
        tr_sel  = train_idx_bal[perm[val_n:]]
        Xtr_df = X_all_df.iloc[tr_sel]
        ytr = y_all[tr_sel]
        Xva_df = X_all_df.iloc[val_sel]
        yva = y_all[val_sel]
        # XGBoost accepts array-like; pass numpy to be explicit for eval
        eval_set = [(Xva_df.to_numpy(dtype="float32"), yva)]
        log(f"Using validation split for early stopping: {len(tr_sel)} train, {len(val_sel)} val")

    # Build monotone constraint vector: +1 on 'qnum', 0 otherwise
    mon = [0] * len(feat_cols)
    if "qnum" in feat_cols:
        mon[feat_cols.index("qnum")] = 1
    mon_str = "(" + ",".join(map(str, mon)) + ")"

    # Define final model (objective = MAE as in optimization)
    model = XGBRegressor(
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
        early_stopping_rounds=args.early_stopping_rounds if eval_set is not None else 0,
        monotone_constraints=mon_str,
    )

    log("Fitting final model on ALL balanced rows...")
    if eval_set is not None:
        model.fit(Xtr_df, ytr, eval_set=eval_set, verbose=False)
    else:
        model.fit(Xtr_df, ytr)

    # Save model(s)
    model_path = args.out_dir / "xgb_model.json"
    model.save_model(model_path.as_posix())
    if joblib is not None:
        try:
            joblib.dump(model, args.out_dir / "xgb_model.joblib")
        except Exception:
            pass
    with open(args.out_dir / "feature_names.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    # Feature importance (5 types)
    booster = model.get_booster()
    try:
        booster.feature_names = feat_cols
    except Exception:
        pass
    imp_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    imp_frames = []
    for it in imp_types:
        scores = booster.get_score(importance_type=it)
        row = {f: float(scores.get(f, 0.0)) for f in feat_cols}
        imp_frames.append(pd.DataFrame({"feature": list(row.keys()), it: list(row.values())}))
    imp_df = imp_frames[0]
    for k in range(1, len(imp_frames)):
        imp_df = imp_df.merge(imp_frames[k], on="feature", how="left")
    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)
    imp_df.insert(0, "rank_gain", np.arange(1, len(imp_df)+1))
    imp_df.to_csv(args.out_dir / "feature_importance.csv", index=False)

    # In-sample predictions (on ALL rows; residuals only where obs finite)
    log("Scoring in-sample predictions for reporting...")
    preds_all = model.predict(X_all_df).astype("float32")
    out_pred = pd.DataFrame({
        "station_id": df["station_id"].astype(str) if "station_id" in df.columns else pd.Series([None]*len(df)),
        "observation_type": df["observation_type"].astype(str),
        "observation": y_all,
        "pred": preds_all,
        "residual": preds_all - y_all,
        "lat": df["lat"] if "lat" in df.columns else pd.Series([np.nan]*len(df)),
        "lon": df["lon"] if "lon" in df.columns else pd.Series([np.nan]*len(df)),
        "height_m": df["height_m"] if "height_m" in df.columns else pd.Series([np.nan]*len(df)),
        "qnum": df["qnum"] if "qnum" in df.columns else pd.Series([np.nan]*len(df)),
    })
    out_pred.to_csv(args.out_dir / "in_sample_predictions.csv", index=False)

    # Training metrics (in-sample — clearly labeled)
    train_metrics_overall = metrics(y_all, preds_all)
    train_metrics_gs  = metrics(y_all[is_gs],  preds_all[is_gs])
    train_metrics_asos= metrics(y_all[is_asos],preds_all[is_asos])
    with open(args.out_dir / "metrics_training.json", "w") as f:
        json.dump({
            "overall_in_sample": train_metrics_overall,
            "GS_in_sample": train_metrics_gs,
            "ASOS_in_sample": train_metrics_asos,
        }, f, indent=2)
    log(f"Training (in-sample) — RMSE={train_metrics_overall['rmse']:.3f}, MAE={train_metrics_overall['mae']:.3f}, N={train_metrics_overall['n']}")

    # Optional SHAP summary (mean/median/p95 of |contrib| per feature)
    if args.shap.lower() == "true":
        if shap is None:
            log("[WARN] SHAP not installed; skipping.")
        else:
            try:
                n_cap = int(args.shap_sample)
                idx = train_idx_bal
                if idx.size > n_cap:
                    idx = np.random.default_rng(args.seed).choice(idx, size=n_cap, replace=False)
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_all_df.iloc[idx], check_additivity=False)
                abs_shap = np.abs(np.asarray(shap_vals))
                stats = {
                    "feature": feat_cols,
                    "mean_abs_shap": abs_shap.mean(axis=0).tolist(),
                    "median_abs_shap": np.median(abs_shap, axis=0).tolist(),
                    "p95_abs_shap": np.percentile(abs_shap, 95, axis=0).tolist(),
                }
                shap_df = pd.DataFrame(stats).sort_values("mean_abs_shap", ascending=False)
                shap_df.to_csv(args.out_dir / "shap_summary.csv", index=False)
                log("Saved SHAP summary (shap_summary.csv).")
            except Exception as e:
                log(f"[WARN] SHAP computation failed: {e}")

    # Metadata for reproducibility
    meta = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
        "infile": str(args.infile),
        "out_dir": str(args.out_dir),
        "seed": int(args.seed),
        "xgb_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
            "n_jobs": args.n_jobs_model,
            "early_stopping_rounds": (args.early_stopping_rounds if (args.val_frac > 0 and args.early_stopping_rounds > 0) else 0),
        },
        "features": feat_cols,
        "wind_features_requested": args.wind_features,
        "aux_features_requested": args.aux_features,
        "include_gwa": bool(args.include_gwa),
        "gwa_file": str(args.gwa_file) if args.gwa_file is not None else None,
        "gwa_col_used": gwa_feature_name,
        "balance_strategy": args.balance_strategy,
        "counts": counts,
        "versions": version_info(),
    }
    with open(args.out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Saved model & artifacts to: {args.out_dir}")
    log("Done.")

if __name__ == "__main__":
    # keep libs from oversubscribing if you distribute runs via processes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    main()
