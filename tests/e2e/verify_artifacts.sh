#!/usr/bin/env bash
# Verify that all large, gitignored artifacts required by the e2e suite exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="${1:-$SCRIPT_DIR/required_artifacts.txt}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[FAIL] Artifact manifest not found: $MANIFEST" >&2
    exit 1
fi

missing=()
while IFS= read -r rel_path || [[ -n "$rel_path" ]]; do
    rel_path="${rel_path%%#*}"
    rel_path="${rel_path#"${rel_path%%[![:space:]]*}"}"
    rel_path="${rel_path%"${rel_path##*[![:space:]]}"}"
    [[ -z "$rel_path" ]] && continue

    if [[ ! -s "$REPO_ROOT/$rel_path" ]]; then
        missing+=("$rel_path")
    fi
done < "$MANIFEST"

if (( ${#missing[@]} > 0 )); then
    echo "[FAIL] Missing required e2e artifacts:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "" >&2
    echo "Restore or symlink these files under $REPO_ROOT before running tests/e2e/run_all.sh." >&2
    exit 1
fi

echo "[PASS] E2E artifact manifest satisfied: $MANIFEST"
