#!/usr/bin/env bash
# helpers.sh — Shared constants, paths, and validation functions for e2e tests.
# Sourced by all stage scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- SSL cert workaround (3DEP API via NREL Netskope proxy) ----
# If a combined CA bundle exists, use it (certifi defaults + NREL Root CA chain).
# Set REQUESTS_CA_BUNDLE before running e2e tests if your network requires it.
if [[ -n "${REQUESTS_CA_BUNDLE:-}" ]]; then
    export SSL_CERT_FILE="$REQUESTS_CA_BUNDLE"
elif [[ -f "$HOME/.ssl/combined_ca_bundle.pem" ]]; then
    export REQUESTS_CA_BUNDLE="$HOME/.ssl/combined_ca_bundle.pem"
    export SSL_CERT_FILE="$HOME/.ssl/combined_ca_bundle.pem"
fi

# ---- Data paths (all repo-internal) ----
DATA_DIR="${REPO_ROOT}/data"

# ---- Reference data for regression checks ----
REF="${REPO_ROOT}/data/reference"

# ---- E2E working directory ----
E2E_DIR="${REPO_ROOT}/data/e2e"

# ---- Colors ----
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

# ---- Timing ----
_STAGE_START=0
_STAGE_NAME=""

stage_start() {
    _STAGE_NAME="$1"
    _STAGE_START=$SECONDS
    echo -e "${CYAN}=== ${_STAGE_NAME} ===${NC}"
}

stage_end() {
    local elapsed=$(( SECONDS - _STAGE_START ))
    echo -e "${GREEN}[PASS]${NC} ${_STAGE_NAME} (${elapsed}s)"
}

# ---- Validation functions ----

check_file() {
    # Usage: check_file FILE
    # Verify file exists and is non-empty.
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}[FAIL]${NC} File not found: $file"
        exit 1
    fi
    if [[ ! -s "$file" ]]; then
        echo -e "${RED}[FAIL]${NC} File is empty: $file"
        exit 1
    fi
    echo "  check_file OK: $(basename "$file")"
}

check_rows() {
    # Usage: check_rows FILE EXPECTED_LINE_COUNT
    # Count lines (including header) and compare to expected.
    local file="$1"
    local expected="$2"
    check_file "$file"
    local actual
    actual=$(wc -l < "$file" | tr -d ' ')
    if [[ "$actual" -ne "$expected" ]]; then
        echo -e "${RED}[FAIL]${NC} Row count mismatch in $(basename "$file"): expected=$expected actual=$actual"
        exit 1
    fi
    echo "  check_rows OK: $actual lines"
}

check_cols() {
    # Usage: check_cols FILE "col1,col2,col3"
    # Verify that named columns exist in the CSV header.
    local file="$1"
    local cols="$2"
    check_file "$file"
    local header
    header=$(head -1 "$file")
    IFS=',' read -ra REQUIRED <<< "$cols"
    for col in "${REQUIRED[@]}"; do
        if ! echo "$header" | grep -q "$col"; then
            echo -e "${RED}[FAIL]${NC} Missing column '$col' in $(basename "$file")"
            echo "  Header: $header"
            exit 1
        fi
    done
    echo "  check_cols OK: ${cols}"
}

check_gzip_cols() {
    # Usage: check_gzip_cols FILE.csv.gz "col1,col2,col3"
    # Verify that named columns exist in a gzipped CSV header.
    local file="$1"
    local cols="$2"
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}[FAIL]${NC} File not found: $file"
        exit 1
    fi
    local header
    header=$(gzip -dc "$file" | head -1)
    IFS=',' read -ra REQUIRED <<< "$cols"
    for col in "${REQUIRED[@]}"; do
        if ! echo "$header" | grep -q "$col"; then
            echo -e "${RED}[FAIL]${NC} Missing column '$col' in $(basename "$file")"
            echo "  Header: $header"
            exit 1
        fi
    done
    echo "  check_gzip_cols OK: ${cols}"
}

check_gzip_rows() {
    # Usage: check_gzip_rows FILE.csv.gz EXPECTED_LINE_COUNT
    # Count lines (including header) in a gzipped CSV and compare to expected.
    local file="$1"
    local expected="$2"
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}[FAIL]${NC} File not found: $file"
        exit 1
    fi
    local actual
    actual=$(gzip -dc "$file" | wc -l | tr -d ' ')
    if [[ "$actual" -ne "$expected" ]]; then
        echo -e "${RED}[FAIL]${NC} Row count mismatch in $(basename "$file"): expected=$expected actual=$actual"
        exit 1
    fi
    echo "  check_gzip_rows OK: $actual lines"
}

diff_csv() {
    # Usage: diff_csv NEW_FILE REF_FILE "col1,col2,col3" [TOLERANCE] [KEY_COLS]
    # Compare numeric columns between two CSVs. Merges on key columns, asserts
    # max absolute difference < tolerance.
    local new_file="$1"
    local ref_file="$2"
    local cols="$3"
    local tol="${4:-1e-6}"
    local keys="${5:-station_id,height_m,qnum}"

    check_file "$new_file"
    check_file "$ref_file"

    python -c "
import pandas as pd, sys

new = pd.read_csv('$new_file')
ref = pd.read_csv('$ref_file')

keys = '$keys'.split(',')
cols = '$cols'.split(',')
tol = float('$tol')

# Use only keys that exist in both
common_keys = [k for k in keys if k in new.columns and k in ref.columns]
if not common_keys:
    print('ERROR: No common key columns found')
    sys.exit(1)

# Merge
merged = new.merge(ref, on=common_keys, suffixes=('_new', '_ref'), how='inner')
if len(merged) == 0:
    print(f'ERROR: No matching rows after merge on {common_keys}')
    sys.exit(1)

errors = []
for col in cols:
    cn = col + '_new' if col + '_new' in merged.columns else col
    cr = col + '_ref' if col + '_ref' in merged.columns else col
    if cn not in merged.columns or cr not in merged.columns:
        # Column might be identical name (not in both with suffix)
        continue
    diff = (merged[cn].astype(float) - merged[cr].astype(float)).abs()
    maxdiff = diff.max()
    if maxdiff > tol:
        errors.append(f'{col}: max_diff={maxdiff:.6e} > tol={tol}')
    else:
        print(f'  diff OK: {col} max_diff={maxdiff:.2e}')

if errors:
    for e in errors:
        print(f'FAIL: {e}')
    sys.exit(1)
"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}[FAIL]${NC} Numeric diff failed for $(basename "$new_file")"
        exit 1
    fi
}
