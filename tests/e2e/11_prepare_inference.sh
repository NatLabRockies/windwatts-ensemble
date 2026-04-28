#!/usr/bin/env bash
# 11_prepare_inference.sh — Stage 6I: Prepare inference table (~15 min).
source "$(dirname "$0")/helpers.sh"

stage_start "11 Prepare Inference (Stage 6I)"

wem-prepare-inference \
  --in  "$E2E_DIR/grid/merged_quantiles_all_with_elev_filled.csv" \
  --out "$E2E_DIR/grid/inference_table.csv"

# Validate
check_file "$E2E_DIR/grid/inference_table.csv"

# Numeric check against reference (sample check on a few columns)
diff_csv \
  "$E2E_DIR/grid/inference_table.csv" \
  "$REF/grid/inference_table.csv" \
  "hrrr,wtk,wtk_led_conus" \
  "1e-6" \
  "lat,lon,height_m,qnum"

stage_end
