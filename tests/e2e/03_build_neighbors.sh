#!/usr/bin/env bash
# 03_build_neighbors.sh — Stage 3C: Build LOOCV neighbor lists (10km radius).
source "$(dirname "$0")/helpers.sh"

stage_start "03 Build Neighbors (Stage 3C)"

wem-build-neighbors \
  --infile  "$E2E_DIR/training/combined_quantiles_long_with_topo.csv" \
  --outfile "$E2E_DIR/training/combined_quantiles_long_with_topo_loocv_10km.csv" \
  --radius-km 10

# Validate
check_rows "$E2E_DIR/training/combined_quantiles_long_with_topo_loocv_10km.csv" 221090
check_cols "$E2E_DIR/training/combined_quantiles_long_with_topo_loocv_10km.csv" \
  "neighbors_10km_site_ids,neighbors_10km_count"

stage_end
