#!/usr/bin/env bash
# 01_merge_cohorts.sh — Stage 3A: Merge ASOS + GS cohorts into long format.
source "$(dirname "$0")/helpers.sh"

stage_start "01 Merge Cohorts (Stage 3A)"

wem-merge-cohorts \
  --obs_asos    "$E2E_DIR/quantiles/asos/obs.csv" \
  --era5_asos   "$E2E_DIR/quantiles/asos/era5.csv" \
  --hrrr_asos   "$E2E_DIR/quantiles/asos/hrrr.csv" \
  --wtk_asos    "$E2E_DIR/quantiles/asos/wtk.csv" \
  --ledc_asos   "$E2E_DIR/quantiles/asos/ledc.csv" \
  --ledclim_asos "$E2E_DIR/quantiles/asos/ledclim.csv" \
  --obs_gs      "$E2E_DIR/quantiles/gs/obs.csv" \
  --era5_gs     "$E2E_DIR/quantiles/gs/era5.csv" \
  --hrrr_gs     "$E2E_DIR/quantiles/gs/hrrr.csv" \
  --wtk_gs      "$E2E_DIR/quantiles/gs/wtk.csv" \
  --ledc_gs     "$E2E_DIR/quantiles/gs/ledc.csv" \
  --ledclim_gs  "$E2E_DIR/quantiles/gs/ledclim.csv" \
  --out "$E2E_DIR/training/combined_quantiles_long.csv"

# Validate: 221,089 data rows + 1 header = 221,090 lines
check_rows "$E2E_DIR/training/combined_quantiles_long.csv" 221090
check_cols "$E2E_DIR/training/combined_quantiles_long.csv" \
  "station_id,height_m,qnum,observation,era5,hrrr,wtk,wtk_led_conus,wtk_led_climate,observation_type"

stage_end
