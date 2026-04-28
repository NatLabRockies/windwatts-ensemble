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

# WTK-LED CONUS (10 m) files — local paths
WTK_LED_FILES=(
  "/datasets/WIND/conus/v2.0.0/2018/conus_2018_10m.h5"
  "/datasets/WIND/conus/v2.0.0/2019/conus_2019_10m.h5"
  "/datasets/WIND/conus/v2.0.0/2020/conus_2020_10m.h5"
)

OUT_CSV="wtk_led_conus_quantiles_2018_2020.csv"

# Batch size (sites per batch)
BATCH_SIZE=100

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

mkdir -p logs
LOG_FILE="logs/wtk_led_conus_qtls_${SLURM_JOB_ID:-manual}.out"

# Mirror stdout+stderr to a log file live
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "────────────────────────────────────────────────────────"
echo "Job ID:        ${SLURM_JOB_ID:-manual}"
echo "Host:          $(hostname)"
echo "Start time:    $(date -Is)"
echo "Sites CSV:     ${SITES_CSV}"
echo "Output CSV:    ${OUT_CSV}"
echo "Batch size:    ${BATCH_SIZE}"
echo "WTK-LED files:"
for f in "${WTK_LED_FILES[@]}"; do
  echo "  - $f"
done
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
  wem-asos-wtkled-conus \
    --sites "${SITES_CSV}" \
    --out   "${OUT_CSV}" \
    --batch "${BATCH_SIZE}" \
    --files "${WTK_LED_FILES[@]}"

echo "────────────────────────────────────────────────────────"
echo "End time:      $(date -Is)"
echo "Done."
