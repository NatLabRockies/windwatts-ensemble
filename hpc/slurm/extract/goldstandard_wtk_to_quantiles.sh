#!/bin/bash
#SBATCH --job-name=wtk_qtls_gs
#SBATCH --partition=short           
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/wtk_qtls_gs_%j.out  
#SBATCH --error=logs/wtk_qtls_gs_%j.out
#SBATCH --account=tap   

set -euo pipefail

# ===== User-configurable paths =====
SITES_CSV="era5_quantiles_gold_standard_2007_2024.csv"
WTK_DATA_DIR="/datasets/WIND/conus/v1.0.0"  # directory with wtk_conus_2007.h5 … 2013
OUT_CSV="wtk_quantiles_gold_standard_2007_2013.csv"

# Batch size & logging cadence (tune to your node)
BATCH_SIZE=100
LOG_EVERY=200

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

mkdir -p logs
LOG_FILE="logs/wtk_qtls_${SLURM_JOB_ID:-manual}.out"

echo "────────────────────────────────────────────────────────"
echo "Job ID:        ${SLURM_JOB_ID:-manual}"
echo "Host:          $(hostname)"
echo "Start time:    $(date -Is)"
echo "Sites CSV:     ${SITES_CSV}"
echo "WTK data dir:  ${WTK_DATA_DIR}"
echo "Output CSV:    ${OUT_CSV}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Log every:     ${LOG_EVERY}"
echo "Python:        $(command -v python || true)"
python -V || true
echo "────────────────────────────────────────────────────────"

# Ensure Python logs are unbuffered so progress appears in real time
export PYTHONUNBUFFERED=1
# Make tqdm update reasonably frequently in non-TTY logs
export TQDM_MININTERVAL=0.5

# ===== Run the job =====
time srun --cpu-bind=none \
  wem-gs-wtk \
    --sites "${SITES_CSV}" \
    --data-dir "${WTK_DATA_DIR}" \
    --out "${OUT_CSV}" \
    --batch-size "${BATCH_SIZE}" \
    --log-every "${LOG_EVERY}"

echo "────────────────────────────────────────────────────────"
echo "End time:      $(date -Is)"
echo "Done."
