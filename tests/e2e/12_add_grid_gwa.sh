#!/usr/bin/env bash
# 12_add_grid_gwa.sh — Stage 6J: Add GWA features to grid inference table (~15 min).
source "$(dirname "$0")/helpers.sh"

stage_start "12 Add Grid GWA (Stage 6J)"

wem-grid-gwa \
  --in   "$E2E_DIR/grid/inference_table.csv" \
  --out  "$E2E_DIR/grid/inference_table_with_gwa.csv" \
  --gwa10  "$E2E_DIR/GWA_wind-speed_10m.tif" \
  --gwa50  "$E2E_DIR/GWA_wind-speed_50m.tif" \
  --gwa100 "$E2E_DIR/GWA_wind-speed_100m.tif" \
  --gwa150 "$E2E_DIR/GWA_wind-speed_150m.tif"

# Validate
check_file "$E2E_DIR/grid/inference_table_with_gwa.csv"

# Numeric check against reference
diff_csv \
  "$E2E_DIR/grid/inference_table_with_gwa.csv" \
  "$REF/grid/inference_table_with_gwa.csv" \
  "gwa_interp" \
  "1e-6" \
  "lat,lon,height_m,qnum"

stage_end
