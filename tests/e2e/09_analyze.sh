#!/usr/bin/env bash
# 09_analyze.sh — Stage 5: Generate analysis outputs (boxplots, metrics).
source "$(dirname "$0")/helpers.sh"

stage_start "09 Analyze (Stage 5)"

wem-analyze \
  --infile "$E2E_DIR/models/loocv/ml_results.csv" \
  --outdir "$E2E_DIR/output/analysis/" \
  --conus \
  --gwa "$E2E_DIR/training/site_height_ws_avg_with_gwa.csv"

# Validate: key outputs exist
check_file "$E2E_DIR/output/analysis/bias_boxplots.png"
check_file "$E2E_DIR/output/analysis/abs_bias_boxplots.png"
check_file "$E2E_DIR/output/analysis/site_metrics_gs.csv"

# Numeric check against reference
diff_csv \
  "$E2E_DIR/output/analysis/site_metrics_gs.csv" \
  "$REF/analysis/site_metrics_gs.csv" \
  "mean_pred_observation,bias_pred_observation" \
  "1e-6" \
  "station_id,height_m"

stage_end
