#!/bin/bash
# Run ONE WTK-LED tile interactively (inside an salloc) and stream logs live.
# Usage (inside an salloc):  ./debug_one_tile_wtkled.sh --tile <TILE_ID>
#        (or) to auto-pick the first tile:  ./debug_one_tile_wtkled.sh
#
# Tip: start an interactive allocation first (example below).

set -euo pipefail

# ---------------- user settings ----------------
ERA5_GRID="era5_grid.csv"
WTKLED_SRC="/datasets/WIND/conus/v2.0.0"
OUT_DIR="out_wtkled_debug"
FORMAT="csv"                               # or "parquet"
WEM_CMD="wem-grid-wtkled"                          # WEM CLI command for WTK-LED grid extraction
CONDA_ENV="${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}" # adjust with WEM_CONDA_ENV if needed
# ------------------------------------------------

# Parse args
TILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tile) TILE="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Make sure we're in an allocation (salloc). Otherwise, tell the user how.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: No Slurm allocation detected."
  echo "Start one, then run this script, e.g.:"
  echo "  salloc -p debug -N 1 -t 00:45:00 -A tap --job-name=wtkled_dbg --mem=0"
  echo "  ./debug_one_tile_wtkled.sh --tile <TILE_ID>"
  exit 1
fi

# Env
ml conda || true
conda activate "$CONDA_ENV"

mkdir -p "$OUT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# If no tile specified, pick the first tile from --list-tiles
if [[ -z "$TILE" ]]; then
  echo "[INFO] No --tile provided; choosing the first available tile from grid…"
  TILE="$(
    $WEM_CMD --era5-grid "$ERA5_GRID" --list-tiles | head -n 1
  )"
  if [[ -z "$TILE" ]]; then
    echo "Could not determine a tile (is $ERA5_GRID valid?)." >&2
    exit 1
  fi
fi

EXT="$FORMAT"; [[ "$FORMAT" == "csv" || "$FORMAT" == "parquet" ]] || EXT="csv"
OUT_FP="${OUT_DIR}/tile_${TILE}.${EXT}"

echo "============================================================"
echo "[RUN] Job $SLURM_JOB_ID  |  TILE=$TILE"
echo "       format=$FORMAT out=$OUT_FP"
echo "============================================================"

# -u = unbuffered; -l = label task output with task id (handy if you add more srun steps)
$WEM_CMD \
    --era5-grid "$ERA5_GRID" \
    --data-dir "$WTKLED_SRC" \
    --tile-only "$TILE" \
    --out-dir "$OUT_DIR" \
    --format "$FORMAT"

echo "------------------------------------------------------------"
echo "[DONE] Tile $TILE → $OUT_FP"
echo "         (timing JSON lines are printed above as TILE_METRICS / SETUP_METRICS)"
