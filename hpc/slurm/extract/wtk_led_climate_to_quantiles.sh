#!/bin/bash
#SBATCH --job-name=wtk_led_conus_qtls
#SBATCH --partition=standard           
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/wtk_qtls_%j.out  
#SBATCH --error=logs/wtk_qtls_%j.out
#SBATCH --account=tap 

set -euo pipefail

# ===== User-configurable paths =====
SITES_CSV="era5_quantiles_2007_2024.csv"
DATA_DIR="/datasets/WIND/ANL_4km_north_america"   # contains north_america_YYYY.h5 (2007..2020)
OUT_CSV="wtk_led_climate_quantiles_2007_2020.csv"

# Batch size (sites per batch) & logging cadence
BATCH_SIZE=100
LOG_EVERY=200

# ===== Environment =====
ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

mkdir -p logs
LOG_FILE="logs/wtk_led_climate_qtls_${SLURM_JOB_ID:-manual}.out"

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
echo "CPUs/task:     ${SLURM_CPUS_PER_TASK:-unset}"
echo "Python:        $(command -v python || true)"
python -V || true
echo "Log file:      ${LOG_FILE}"
echo "────────────────────────────────────────────────────────"

# Ensure Python prints progress in real time
export PYTHONUNBUFFERED=1
# Make tqdm update reasonably frequently in non-TTY logs
export TQDM_MININTERVAL=0.5

# Quick sanity echo (files presence is validated inside the script too)
echo "Expecting files: ${DATA_DIR}/north_america_{2007..2020}.h5"

# ===== Run the job =====
time srun --cpu-bind=none \
  wem-asos-wtkled-climate \
    --sites    "${SITES_CSV}" \
    --out      "${OUT_CSV}" \
    --data-dir "${DATA_DIR}" \
    --batch    "${BATCH_SIZE}" \
    --log-every "${LOG_EVERY}"

echo "────────────────────────────────────────────────────────"
echo "End time:      $(date -Is)"
echo "Done."
