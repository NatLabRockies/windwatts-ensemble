#!/usr/bin/env bash
# 13_infer.sh — Stage 6K: Run XGBoost inference on full grid (~30 min).
source "$(dirname "$0")/helpers.sh"

stage_start "13 Infer (Stage 6K)"

wem-infer \
  --in "$E2E_DIR/grid/inference_table_with_gwa.csv" \
  --model-dir "$E2E_DIR/models/final/" \
  --out "$E2E_DIR/grid/predictions_fullgrid.csv" \
  --overwrite

# Validate
check_file "$E2E_DIR/grid/predictions_fullgrid.csv"
check_cols "$E2E_DIR/grid/predictions_fullgrid.csv" "pred_observation"

stage_end
