#!/usr/bin/env bash
# 04_obs_wsavg.sh — Stage 3D: Compute observed wind speed averages per site-height.
source "$(dirname "$0")/helpers.sh"

stage_start "04 Obs Wind Speed Average (Stage 3D)"

wem-obs-wsavg \
  --in  "$E2E_DIR/training/combined_quantiles_long_with_topo_loocv_10km.csv" \
  --out "$E2E_DIR/training/site_height_ws_avg.csv"

# Validate
check_file "$E2E_DIR/training/site_height_ws_avg.csv"
check_cols "$E2E_DIR/training/site_height_ws_avg.csv" "station_id,height_m,ws_avg"

# Numeric check against reference
diff_csv \
  "$E2E_DIR/training/site_height_ws_avg.csv" \
  "$REF/training/site_height_ws_avg.csv" \
  "ws_avg" \
  "1e-6" \
  "station_id,height_m"

stage_end
