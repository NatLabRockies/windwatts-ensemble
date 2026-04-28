#!/usr/bin/env bash
# 05_add_gwa.sh — Stage 3E: Add Global Wind Atlas features.
source "$(dirname "$0")/helpers.sh"

stage_start "05 Add GWA (Stage 3E)"

wem-add-gwa \
  --in   "$E2E_DIR/training/site_height_ws_avg.csv" \
  --out  "$E2E_DIR/training/site_height_ws_avg_with_gwa.csv" \
  --gwa10  "$E2E_DIR/GWA_wind-speed_10m.tif" \
  --gwa50  "$E2E_DIR/GWA_wind-speed_50m.tif" \
  --gwa100 "$E2E_DIR/GWA_wind-speed_100m.tif" \
  --gwa150 "$E2E_DIR/GWA_wind-speed_150m.tif"

# Validate
check_file "$E2E_DIR/training/site_height_ws_avg_with_gwa.csv"
check_cols "$E2E_DIR/training/site_height_ws_avg_with_gwa.csv" \
  "gwa_10,gwa_50,gwa_100,gwa_150,gwa_interp"

# Numeric check against reference
diff_csv \
  "$E2E_DIR/training/site_height_ws_avg_with_gwa.csv" \
  "$REF/training/site_height_ws_avg_with_gwa.csv" \
  "gwa_interp" \
  "1e-6" \
  "station_id,height_m"

stage_end
