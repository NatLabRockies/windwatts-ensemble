#!/usr/bin/env bash
# 08_train_final.sh — Stage 4B: Train final model on all data (~2 min).
source "$(dirname "$0")/helpers.sh"

stage_start "08 Train Final Model (Stage 4B)"

wem-train-final \
  --infile  "$E2E_DIR/training/combined_quantiles_long_with_topo_loocv_10km.csv" \
  --out-dir "$E2E_DIR/models/final/" \
  --n_jobs_model 8 --shap false \
  --gwa-file "$E2E_DIR/training/site_height_ws_avg_with_gwa.csv" \
  --gwa-col gwa_interp --include-gwa

# Validate: model artifacts exist
check_file "$E2E_DIR/models/final/xgb_model.json"
check_file "$E2E_DIR/models/final/feature_names.json"
check_file "$E2E_DIR/models/final/metrics_training.json"

stage_end
