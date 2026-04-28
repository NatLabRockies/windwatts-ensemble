#!/bin/bash
# Run WTK-LED tiles locally with N-way parallelism (no Slurm).
# Usage: ./run_tiles_locally_wtkled.sh tiles_wtkled.txt 10
# Defaults: 10 tiles at a time, 8 threads per tile (good for ~104 cores).

set -euo pipefail

TILES_FILE="${1:-tiles_wtkled.txt}"
PARALLEL_JOBS="${2:-10}"                    # how many tiles to run at once
THREADS_PER_TILE="${THREADS_PER_TILE:-10}"   # cores per tile (export to override)

# ---- user settings ------------------------------------------------
ERA5_GRID="era5_grid.csv"
WTKLED_SRC="/datasets/WIND/conus/v2.0.0"      # per-year subdirs with conus_<year>_<h>m.h5
OUT_DIR="out_wtkled"
FORMAT="csv"                                  # or "parquet"
WEM_CMD="wem-grid-wtkled"
PROJECTION="${PROJECTION:-auto}"              # auto|dense|sparse
DENSE_MAX_MB="${DENSE_MAX_MB:-64}"            # max RAM for densifying W per tile
EXTRA_ARGS=""                                  # e.g. '--heights 30,40,60,80,100 --tile-km 250'
OVERWRITE=0                                    # set to 1 to pass --overwrite
# -------------------------------------------------------------------

mkdir -p logs "${OUT_DIR}"
STATUS_DIR="logs/status"
mkdir -p "$STATUS_DIR"

# (Optional) Activate your env
ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

# Threading: give each Python process THREADS_PER_TILE cores
export OMP_NUM_THREADS="${THREADS_PER_TILE}"
export MKL_NUM_THREADS="${THREADS_PER_TILE}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_TILE}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_TILE}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE


# Info banner
total_threads=$(( PARALLEL_JOBS * THREADS_PER_TILE ))
echo "[INFO] PARALLEL_JOBS=${PARALLEL_JOBS}, THREADS_PER_TILE=${THREADS_PER_TILE}, total_threads=${total_threads}"

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

  # Use actual tile id with required prefix for logs/flags
  local base="tile_wtkled_${tile}"

  local out_fp="${OUT_DIR}/tile_${tile}.${ext}"
  local log_out="logs/${base}.out"
  local log_err="logs/${base}.err"
  local ok_flag="${STATUS_DIR}/${base}.ok"
  local fail_flag="${STATUS_DIR}/${base}.fail"

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
        --data-dir "$WTKLED_SRC"
        --tile-only "$tile"
        --out-dir "$OUT_DIR"
        --format "$FORMAT"
        --num-threads "$THREADS_PER_TILE"
        --projection "$PROJECTION"
        --dense-max-mb "$DENSE_MAX_MB" )
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
fails=$(ls "$STATUS_DIR"/tile_wtkled_*.fail 2>/dev/null | wc -l | tr -d ' ')
oks=$(ls "$STATUS_DIR"/tile_wtkled_*.ok 2>/dev/null | wc -l | tr -d ' ')
echo "=================================================="
echo "Completed OK: $oks   Failed: $fails   Total: $((oks + fails))"
if (( fails > 0 )); then
  echo "Failed tiles:"
  sed 's#^.*/##; s/\.fail$//' "$STATUS_DIR"/tile_wtkled_*.fail 2>/dev/null | sort -n
fi
