#!/usr/bin/env bash
# 10_fill_missing.sh — Stage 6H: Fill missing elevation values in grid data.
source "$(dirname "$0")/helpers.sh"

stage_start "10 Fill Missing (Stage 6H)"

wem-fill-missing \
  --in  "$E2E_DIR/grid/merged_quantiles_all_with_elev.csv" \
  --out "$E2E_DIR/grid/merged_quantiles_all_with_elev_filled.csv"

# Validate
check_file "$E2E_DIR/grid/merged_quantiles_all_with_elev_filled.csv"

# Numeric check against reference (wide format — use grid_id,height_m as keys)
diff_csv \
  "$E2E_DIR/grid/merged_quantiles_all_with_elev_filled.csv" \
  "$REF/grid/merged_quantiles_all_with_elev_filled.csv" \
  "elevation_m" \
  "1e-6" \
  "grid_id,height_m"

stage_end
