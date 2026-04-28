#!/bin/bash
# Split a tile list into N shard files (round-robin), skipping blanks/comments.
# Usage: ./split_missing_tiles_wtkled.sh missing_tiles_wtkled.txt 8 splits_wtkled

set -euo pipefail

IN="${1:-missing_tiles_wtkled.txt}"
PARTS="${2:-8}"
OUTDIR="${3:-splits_wtkled_v2}"

if [[ ! -f "$IN" ]]; then
  echo "Input file not found: $IN" >&2
  exit 1
fi
if ! [[ "$PARTS" =~ ^[0-9]+$ ]] || (( PARTS < 1 )); then
  echo "Parts must be a positive integer (got: $PARTS)" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

# Read tiles safely: trim, skip blanks and lines starting with '#'
mapfile -t TILES < <(awk '
  BEGIN{RS="\n"}
  /^[[:space:]]*(#|$)/ {next}
  {gsub(/\r/,""); gsub(/^[[:space:]]+|[[:space:]]+$/,""); print}
' "$IN")

if (( ${#TILES[@]} == 0 )); then
  echo "No tiles found in $IN (after filtering blanks/comments)." >&2
  exit 1
fi

# Initialize/clear output files
for p in $(seq -f "%02g" 1 "$PARTS"); do
  : > "$OUTDIR/missing_tiles_wtkled_part_${p}.txt"
done

# Round-robin assignment for better balance across nodes
for i in "${!TILES[@]}"; do
  idx=$(( i % PARTS + 1 ))
  suf=$(printf "%02d" "$idx")
  echo "${TILES[$i]}" >> "$OUTDIR/missing_tiles_wtkled_part_${suf}.txt"
done

# Summary
echo "Wrote shards to: $OUTDIR"
for p in $(seq -f "%02g" 1 "$PARTS"); do
  fp="$OUTDIR/missing_tiles_wtkled_part_${p}.txt"
  count=$(grep -cve '^[[:space:]]*$' "$fp" || true)
  echo "  part ${p}: ${count} tiles  ->  ${fp}"
done

