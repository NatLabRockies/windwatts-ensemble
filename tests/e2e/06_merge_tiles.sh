#!/usr/bin/env bash
# 06_merge_tiles.sh — Stage 6M: Merge per-tile grid outputs into per-height CSVs.
# Uses synthetic tile data (no real HPC data needed).
source "$(dirname "$0")/helpers.sh"

stage_start "06 Merge Tiles (Stage 6M)"

TILE_DIR="$E2E_DIR/grid/tiles/test"
MERGED_DIR="$E2E_DIR/grid/merged/test"

# ---- Generate 3 synthetic tile CSVs with 2 heights (60, 100) ----
python -c "
import pandas as pd, os

tiles = {
    'tile_001.csv': [
        {'grid_id': 'G001', 'lat': 40.0, 'lon': -105.0, 'height_m': 60, 'q000': 0.0, 'q050': 5.0, 'q100': 12.0},
        {'grid_id': 'G001', 'lat': 40.0, 'lon': -105.0, 'height_m': 100, 'q000': 0.0, 'q050': 6.5, 'q100': 14.0},
    ],
    'tile_002.csv': [
        {'grid_id': 'G002', 'lat': 41.0, 'lon': -104.0, 'height_m': 60, 'q000': 0.1, 'q050': 4.8, 'q100': 11.5},
        {'grid_id': 'G002', 'lat': 41.0, 'lon': -104.0, 'height_m': 100, 'q000': 0.1, 'q050': 6.2, 'q100': 13.5},
    ],
    'tile_003.csv': [
        {'grid_id': 'G003', 'lat': 42.0, 'lon': -103.0, 'height_m': 60, 'q000': 0.2, 'q050': 5.5, 'q100': 13.0},
        {'grid_id': 'G003', 'lat': 42.0, 'lon': -103.0, 'height_m': 100, 'q000': 0.2, 'q050': 7.0, 'q100': 15.0},
    ],
}

for fname, rows in tiles.items():
    pd.DataFrame(rows).to_csv(os.path.join('${TILE_DIR}', fname), index=False)

print(f'  Created {len(tiles)} synthetic tile files')
"

# ---- Run merge ----
wem-merge-tiles \
  --in-dir  "$TILE_DIR" \
  --out-dir "$MERGED_DIR" \
  --prefix  test \
  --dedupe \
  --overwrite

# ---- Validate outputs ----
check_file "$MERGED_DIR/test_quantiles_60m.csv"
check_file "$MERGED_DIR/test_quantiles_100m.csv"

check_cols "$MERGED_DIR/test_quantiles_60m.csv"  "grid_id,lat,lon,height_m,q000,q050,q100"
check_cols "$MERGED_DIR/test_quantiles_100m.csv" "grid_id,lat,lon,height_m,q000,q050,q100"

# 3 data rows + 1 header = 4 lines per height
check_rows "$MERGED_DIR/test_quantiles_60m.csv"  4
check_rows "$MERGED_DIR/test_quantiles_100m.csv" 4

stage_end
