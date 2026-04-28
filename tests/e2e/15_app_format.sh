#!/usr/bin/env bash
# 15_app_format.sh — Stage 7C: Convert grid predictions to per-location app files.
source "$(dirname "$0")/helpers.sh"

stage_start "15 App Format (Stage 7C)"

# Use a small subset (first 1001 lines = header + 1000 rows) to keep runtime short
head -1001 "$E2E_DIR/grid/site_quantiles_predicted.csv" > "$E2E_DIR/grid/app_format_input.csv"

wem-app-format \
  --in "$E2E_DIR/grid/app_format_input.csv" \
  --out-dir "$E2E_DIR/output/app" \
  --make-index \
  --skip-missing

# Validate location index was created
check_file "$E2E_DIR/output/app/location_index.csv.gz"

# Verify at least 1 location .csv.gz file was created (besides the index)
LOC_COUNT=$(find "$E2E_DIR/output/app" -name "??????.csv.gz" | wc -l | tr -d ' ')
if [[ "$LOC_COUNT" -lt 1 ]]; then
    echo -e "${RED}[FAIL]${NC} No location files created (expected at least 1)"
    exit 1
fi
echo "  Location files created: $LOC_COUNT"

# Spot-check one location file: verify columns and row count
FIRST_LOC=$(find "$E2E_DIR/output/app" -name "??????.csv.gz" | head -1)
check_gzip_cols "$FIRST_LOC" "probability,windspeed_30m,windspeed_40m,windspeed_50m,windspeed_60m,windspeed_80m,windspeed_100m"
check_gzip_rows "$FIRST_LOC" 102  # header + 101 quantile rows

stage_end
