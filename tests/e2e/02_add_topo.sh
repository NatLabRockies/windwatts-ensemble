#!/usr/bin/env bash
# 02_add_topo.sh — Stage 3B: Add topography from reference data.
#
# Merges elevation/slope/aspect columns from the reference file onto the
# current input using station_id as the join key. This ensures deterministic
# regression testing — the live 3DEP API is tested by the production pipeline,
# not the e2e suite.
source "$(dirname "$0")/helpers.sh"

stage_start "02 Add Topography (Stage 3B)"

INPUT="$E2E_DIR/training/combined_quantiles_long.csv"
OUTPUT="$E2E_DIR/training/combined_quantiles_long_with_topo.csv"
REFERENCE="$REF/training/combined_quantiles_long_with_topo.csv"

FALLBACK_MERGE='
import pandas as pd, sys
inp = pd.read_csv(sys.argv[1], low_memory=False)
ref = pd.read_csv(sys.argv[2], low_memory=False)
# Extract unique topo per station_id (all rows for a station share same topo)
topo_cols = ["station_id", "elevation_m", "slope_deg", "aspect_deg"]
ref_topo = ref[topo_cols].drop_duplicates(subset=["station_id"])
merged = inp.merge(ref_topo, on="station_id", how="left")
n_ok = merged.elevation_m.notna().sum()
print(f"  Merged topo from reference: {n_ok}/{len(merged)} rows with elevation")
n_nan = merged.elevation_m.isna().sum()
if n_nan > 0:
    missing = merged[merged.elevation_m.isna()].station_id.unique()
    print(f"  {n_nan} rows ({len(missing)} stations) without elevation (new coords not in reference)")
merged.to_csv(sys.argv[3], index=False)
'

# Always merge topo from reference for deterministic regression testing.
python -c "$FALLBACK_MERGE" "$INPUT" "$REFERENCE" "$OUTPUT"

# Validate
check_rows "$OUTPUT" 221090
check_cols "$OUTPUT" "elevation_m,slope_deg,aspect_deg"

stage_end
