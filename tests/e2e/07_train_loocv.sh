#!/usr/bin/env bash
# 07_train_loocv.sh — Stage 4A: LOOCV XGBoost training (~10 min).
source "$(dirname "$0")/helpers.sh"

stage_start "07 Train LOOCV (Stage 4A)"

wem-train-loocv \
  --infile  "$E2E_DIR/training/combined_quantiles_long_with_topo_loocv_10km.csv" \
  --outfile "$E2E_DIR/models/loocv/ml_results.csv" \
  --balance_strategy downsample \
  --n_jobs_outer 12 --n_jobs_model 1 \
  --gwa-file "$E2E_DIR/training/site_height_ws_avg_with_gwa.csv" \
  --gwa-col gwa_interp --include-gwa

# Validate
check_file "$E2E_DIR/models/loocv/ml_results.csv"
check_cols "$E2E_DIR/models/loocv/ml_results.csv" "pred_observation"

# Numeric check against reference
diff_csv \
  "$E2E_DIR/models/loocv/ml_results.csv" \
  "$REF/loocv/ml_results.csv" \
  "pred_observation" \
  "1e-6"

stage_end
