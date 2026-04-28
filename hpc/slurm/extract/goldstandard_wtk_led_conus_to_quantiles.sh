#!/bin/bash
#SBATCH --job-name=gs_wtk_led_conus_qtls
#SBATCH --partition=standard           
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/gs_wtk_led_conus_qtls_%j.out  
#SBATCH --error=logs/gs_wtk_led_conus_qtls_%j.out
#SBATCH --account=tap   

set -euo pipefail

# ===== User-configurable paths =====
SITES_CSV="era5_quantiles_gold_standard_2007_2024.csv"
DATA_DIR="/datasets/WIND/conus/v2.0.0"          # per-year subdirs with conus_<year>_10m.h5
OUT_CSV="wtk_led_conus_quantiles_gold_standard_2018_2020.csv"

# Batch size (sites per batch) & logging cadence
BATCH_SIZE=100
LOG_EVERY=200

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

mkdir -p logs
LOG_FILE="logs/gs_wtk_led_conus_qtls_${SLURM_JOB_ID:-manual}.out"

# Mirror stdout+stderr to a log file live
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "────────────────────────────────────────────────────────"
echo "Job ID:        ${SLURM_JOB_ID:-manual}"
echo "Host:          $(hostname)"
echo "Start time:    $(date -Is)"
echo "Sites CSV:     ${SITES_CSV}"
echo "Output CSV:    ${OUT_CSV}"
echo "Batch size:    ${BATCH_SIZE}"
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
  wem-gs-wtkled-conus \
    --sites "${SITES_CSV}" \
    --data-dir "${DATA_DIR}" \
    --out "${OUT_CSV}" \
    --batch-size "${BATCH_SIZE}" \
    --log-every "${LOG_EVERY}"

echo "────────────────────────────────────────────────────────"
echo "End time:      $(date -Is)"
echo "Done."
