#!/usr/bin/env python3
"""
Run the final XGB model on the full-grid inference table.

Input table (CSV/Parquet):
  columns: lat, lon, elevation_m, height_m, qnum, wtk, hrrr, wtk_led_conus
  (exact names as produced by prepare_inference.py)

Model artifacts (in --model-dir):
  - Model: model.json / xgb_model.json / final_model.json (XGBRegressor saved_model)
  - Features: features.json / feature_names.json / features_used.json (list or {"features":[...]})
  - Fallback if no features file: use default feature order:
      ['qnum','hrrr','wtk','wtk_led_conus','lat','lon','height_m','elevation_m','gwa_interp']

Output:
  - Same rows and columns as input + a new column 'pred_observation'

Usage:
  python infer_xgb.py \\
    --in inference_table.parquet \\
    --model-dir model_final_all \\
    --out predictions_fullgrid.parquet
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# xgboost sklearn wrapper
from xgboost import XGBRegressor

from wem.utils.logging import log
from wem.utils.io import read_table, write_table


# ----------------------------- model utils -----------------------------
def load_feature_list(model_dir: Path) -> Optional[List[str]]:
    """Try a few common filenames/formats for the saved feature list."""
    for name in ["features.json", "feature_names.json", "features_used.json", "feature_list.json", "features_used.txt"]:
        p = model_dir / name
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".txt":
                with p.open("r") as f:
                    feats = [ln.strip() for ln in f if ln.strip()]
            else:
                js = json.loads(p.read_text())
                if isinstance(js, list):
                    feats = js
                elif isinstance(js, dict) and "features" in js and isinstance(js["features"], list):
                    feats = js["features"]
                else:
                    continue
            feats = [str(c) for c in feats]
            if feats:
                return feats
        except Exception:
            pass
    return None


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Add any missing columns as NaN and return final ordered list."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log(f"[WARN] Missing feature columns in input: {missing} -- filling with NaN.")
        for c in missing:
            df[c] = np.nan
    return cols


# ----------------------------- inference -------------------------------
def run_inference(
    df: pd.DataFrame,
    model_path: Path,
    features: List[str],
    batch_size: int = 500_000,
) -> np.ndarray:
    """Load model.json into XGBRegressor and predict in batches; returns predictions."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = XGBRegressor()
    model.load_model(str(model_path))

    # Ensure numeric float32 matrix in the proper order
    X = df[features].astype("float32")

    n = len(X)
    preds = np.empty(n, dtype=np.float32)

    if n == 0:
        return preds

    # Batch to keep memory/threads happy on big tables
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        preds[start:stop] = model.predict(X.iloc[start:stop].to_numpy(copy=False)).astype(np.float32)

    return preds


def main():
    ap = argparse.ArgumentParser(description="Use the final XGB model to predict on the full-grid table.")
    ap.add_argument("--in",  dest="infile",  type=Path, required=True, help="Inference table (CSV/Parquet).")
    ap.add_argument("--model-dir", type=Path, required=True, help="Folder with model.json and optional features.json.")
    ap.add_argument("--out", dest="outfile", type=Path, required=True, help="Output file (CSV/Parquet) with predictions.")
    ap.add_argument("--batch-size", type=int, default=500_000, help="Rows per prediction batch.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.outfile.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.outfile} (use --overwrite).")

    log(f"Loading input: {args.infile}")
    df = read_table(args.infile)

    # Default feature order (matches training you described)
    default_feats = ["qnum", "hrrr", "wtk", "wtk_led_conus", "lat", "lon", "height_m", "elevation_m", "gwa_interp"]

    # Load feature list if provided; otherwise, fall back
    feats = load_feature_list(args.model_dir) or default_feats
    feats = ensure_columns(df, feats)
    log(f"Using {len(feats)} features: {feats}")

    # Model path — try several common filenames
    model_json = None
    for mname in ["model.json", "xgb_model.json", "final_model.json"]:
        candidate = args.model_dir / mname
        if candidate.exists():
            model_json = candidate
            break
    if model_json is None:
        raise FileNotFoundError(f"No model file found in {args.model_dir} (tried model.json, xgb_model.json, final_model.json)")
    log(f"Loading model: {model_json}")

    # Predict
    log("Running predictions ...")
    preds = run_inference(df, model_json, feats, batch_size=max(1, int(args.batch_size)))

    # Attach and save
    out = df.copy()
    out["pred_observation"] = preds

    log(f"Writing output -> {args.outfile} (rows={len(out):,})")
    write_table(out, args.outfile)
    log("Done.")


if __name__ == "__main__":
    main()
