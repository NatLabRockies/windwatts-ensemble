#!/usr/bin/env bash
# 14_pivot_predictions.sh — Stage 6L: Pivot predictions to wide format (~10 min).
source "$(dirname "$0")/helpers.sh"

stage_start "14 Pivot Predictions (Stage 6L)"

wem-pivot-predictions \
  --in "$E2E_DIR/grid/predictions_fullgrid.csv" \
  --era5 "$E2E_DIR/grid/era5_location_data.csv" \
  --out "$E2E_DIR/grid/site_quantiles_predicted.csv" \
  --decimals 2

# Validate
check_file "$E2E_DIR/grid/site_quantiles_predicted.csv"

stage_end
