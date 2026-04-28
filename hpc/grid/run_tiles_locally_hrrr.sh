#!/bin/bash
# Run HRRR tiles locally with N-way parallelism (no Slurm).
# Usage: ./run_tiles_locally_hrrr.sh tiles_hrrr.txt 20

set -euo pipefail

TILES_FILE="${1:-tiles_hrrr.txt}"
PARALLEL_JOBS="${2:-20}"   # how many tiles to run at once

# ---- user settings ------------------------------------------------
ERA5_GRID="era5_grid.csv"
HRRR_SRC="/datasets/WIND/HRRR"
OUT_DIR="out_hrrr"
FORMAT="csv"               # or "parquet"
WEM_CMD="wem-grid-hrrr"
EXTRA_ARGS=""              # e.g. '--heights 30,40,60,80,100 --tile-km 250'
OVERWRITE=0                # set to 1 to pass --overwrite
# -------------------------------------------------------------------

mkdir -p logs "${OUT_DIR}"
STATUS_DIR="logs/status"
mkdir -p "$STATUS_DIR"

# Activate your env (comment out if not using environment modules)
ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

# Avoid oversubscription when many jobs run on one node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
# Flush logs promptly for tail -f
export PYTHONUNBUFFERED=1

# Read tiles safely: keep newlines, trim, skip blanks/comments
mapfile -t TILES < <(awk '
  BEGIN{RS="\n"}
  /^[[:space:]]*(#|$)/ {next}
  {gsub(/\r/,""); gsub(/^[[:space:]]+|[[:space:]]+$/,""); print}
' "$TILES_FILE")

if (( ${#TILES[@]} == 0 )); then
  echo "No tiles found in $TILES_FILE"
  exit 0
fi

ext="$FORMAT"; [[ "$FORMAT" == "csv" || "$FORMAT" == "parquet" ]] || ext="csv"

# Kill all children on Ctrl-C/TERM
trap 'echo "[INFO] stopping…"; jobs -p | xargs -r kill; wait || true' INT TERM

run_one() {
  local tile="$1"

  # short hash for log/flag names; outputs still use full tile ID
  local sid
  sid=$(printf '%s' "$tile" | sha1sum | awk '{print substr($1,1,12)}')

  local out_fp="${OUT_DIR}/tile_${tile}.${ext}"
  local log_out="logs/tile_${sid}.out"
  local log_err="logs/tile_${sid}.err"
  local ok_flag="${STATUS_DIR}/${sid}.ok"
  local fail_flag="${STATUS_DIR}/${sid}.fail"

  # Skip if output already exists
  if [[ -s "$out_fp" ]]; then
    echo "[SKIP] ${tile} (exists: ${out_fp})" | tee -a "$log_out"
    : > "$ok_flag"
    return 0
  fi

  # Build command safely (EXTRA_ARGS may contain quoted tokens)
  local -a cmd
  cmd=( $WEM_CMD
        --era5-grid "$ERA5_GRID"
        --data-dir "$HRRR_SRC"
        --tile-only "$tile"
        --out-dir "$OUT_DIR"
        --format "$FORMAT" )
  if (( OVERWRITE )); then
    cmd+=( --overwrite )
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args=( ${EXTRA_ARGS} )
    cmd+=( "${extra_args[@]}" )
  fi

  echo "[RUN ] tile=${tile} → ${out_fp}"
  if stdbuf -oL -eL "${cmd[@]}" >"$log_out" 2>"$log_err"; then
    : > "$ok_flag"
    echo "[DONE] ${tile}"
  else
    : > "$fail_flag"
    echo "[FAIL] ${tile} (see $log_err)"
    return 1
  fi
}

# ----- portable concurrency throttle (no wait -n required) -----
for tile in "${TILES[@]}"; do
  run_one "$tile" &
  # throttle while we have >= PARALLEL_JOBS running
  while (( $(jobs -r -p | wc -l) >= PARALLEL_JOBS )); do
    sleep 0.5
  done
done

# Wait for remaining jobs
wait || true

# Summary
fails=$(ls "$STATUS_DIR"/*.fail 2>/dev/null | wc -l | tr -d ' ')
oks=$(ls "$STATUS_DIR"/*.ok 2>/dev/null | wc -l | tr -d ' ')
echo "=================================================="
echo "Completed OK: $oks   Failed: $fails   Total: $((oks + fails))"
if (( fails > 0 )); then
  echo "Failed tiles:"
  sed 's#^.*/##; s/\.fail$//' "$STATUS_DIR"/*.fail 2>/dev/null | sort -n
fi
