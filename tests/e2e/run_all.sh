#!/usr/bin/env bash
# run_all.sh — Run all e2e test stages in sequence.
# Stops on first failure. Reports timing per stage and total.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

STAGES=(
    00_setup
    01_merge_cohorts
    02_add_topo
    03_build_neighbors
    04_obs_wsavg
    05_add_gwa
    06_merge_tiles
    07_train_loocv
    08_train_final
    09_analyze
    10_fill_missing
    11_prepare_inference
    12_add_grid_gwa
    13_infer
    14_pivot_predictions
    15_app_format
)

TOTAL_START=$SECONDS
RESULTS=()

for stage in "${STAGES[@]}"; do
    start=$SECONDS
    echo ""
    if bash "$DIR/${stage}.sh"; then
        elapsed=$(( SECONDS - start ))
        RESULTS+=("${GREEN}[PASS]${NC} ${stage} (${elapsed}s)")
    else
        elapsed=$(( SECONDS - start ))
        RESULTS+=("${RED}[FAIL]${NC} ${stage} (${elapsed}s)")
        echo ""
        echo -e "${RED}=== FAILED at ${stage} ===${NC}"
        echo ""
        echo "Summary so far:"
        for r in "${RESULTS[@]}"; do
            echo -e "  $r"
        done
        exit 1
    fi
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
echo ""
echo -e "${CYAN}=== E2E Test Summary ===${NC}"
for r in "${RESULTS[@]}"; do
    echo -e "  $r"
done
echo ""
echo -e "${GREEN}All ${#STAGES[@]} stages passed in ${TOTAL_ELAPSED}s.${NC}"
