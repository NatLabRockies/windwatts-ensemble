#!/bin/bash
#SBATCH --job-name=hrrr_qtls
#SBATCH --partition=standard           
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/hrrr_qtls_%j.out  
#SBATCH --error=logs/hrrr_qtls_%j.out
#SBATCH --account=tap   

set -euo pipefail

# ===== User-configurable paths =====
SITES_CSV="era5_quantiles_2007_2024.csv"
DATA_DIR="/datasets/WIND/HRRR"                 # contains bchrrr_conus_YYYY.h5
OUT_CSV="hrrr_quantiles_2015_2023.csv"

# Batch size & logging cadence (sites per batch / status print frequency)
BATCH_SIZE=100
LOG_EVERY=200

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

mkdir -p logs
LOG_FILE="logs/hrrr_qtls_${SLURM_JOB_ID:-manual}.out"

# Mirror stdout+stderr to a log file live
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "────────────────────────────────────────────────────────"
echo "Job ID:        ${SLURM_JOB_ID:-manual}"
echo "Host:          $(hostname)"
echo "Start time:    $(date -Is)"
echo "Sites CSV:     ${SITES_CSV}"
echo "Data dir:      ${DATA_DIR}"
echo "Output CSV:    ${OUT_CSV}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Log every:     ${LOG_EVERY}"
echo "Python:        $(command -v python || true)"
python -V || true
echo "Log file:      ${LOG_FILE}"
echo "────────────────────────────────────────────────────────"

# Ensure Python logs are unbuffered so progress appears in real time
export PYTHONUNBUFFERED=1
# Make tqdm update reasonably frequently in non-TTY logs
export TQDM_MININTERVAL=0.5

# ===== Run the job =====
time srun --cpu-bind=none \
  wem-asos-hrrr \
    --sites "${SITES_CSV}" \
    --data-dir "${DATA_DIR}" \
    --out "${OUT_CSV}" \
    --batch "${BATCH_SIZE}" \
    --log-every "${LOG_EVERY}"

echo "────────────────────────────────────────────────────────"
echo "End time:      $(date -Is)"
echo "Done."
