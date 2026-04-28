#!/bin/bash
#SBATCH --job-name=wtkled_tiles
#SBATCH --output=logs/wtkled_tiles_%A_%a.out
#SBATCH --error=logs/wtkled_tiles_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --account=tap

set -euo pipefail

# ---- user settings -----------------------------------------------------------
ERA5_GRID="era5_grid.csv"                    # path to your ERA5 grid (grid_id,lat,lon)
WTKLED_SRC="/datasets/WIND/conus/v2.0.0"     # shared filesystem base dir
OUT_DIR="out_wtkled"                         # per-tile outputs written here
FORMAT="csv"                                 # parquet|csv
TILES_FILE="missing_tiles.txt"                       # one tile ID per line
WEM_CMD="wem-grid-wtkled"                     # WEM CLI command for WTK-LED grid extraction
# ------------------------------------------------------------------------------

mkdir -p logs "$OUT_DIR"
cd "$SLURM_SUBMIT_DIR"

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

# Resolve the tile ID for this array task
if [[ ! -f "$TILES_FILE" ]]; then
  echo "ERROR: $TILES_FILE not found in $PWD" >&2
  exit 1
fi
TILE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TILES_FILE" | tr -d ' \t\r\n')"
if [[ -z "${TILE:-}" ]]; then
  echo "ERROR: No tile for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
echo "[INFO] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID → TILE=$TILE"

# Run the tile directly against the shared dataset path
srun -u $WEM_CMD \
  --era5-grid "$ERA5_GRID" \
  --data-dir "$WTKLED_SRC" \
  --tile-only "$TILE" \
  --out-dir "$OUT_DIR" \
  --format "$FORMAT"

echo "[INFO] Tile $TILE completed."
