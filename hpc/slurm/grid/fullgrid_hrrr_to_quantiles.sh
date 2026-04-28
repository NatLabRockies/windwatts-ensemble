#!/bin/bash
#SBATCH --job-name=hrrrq
#SBATCH --output=hrrr_logs/hrrrq_%j.out
#SBATCH --error=hrrr_logs/hrrrq_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --account=tap
#SBATCH --nodes=1

set -euo pipefail
mkdir -p hrrr_logs
cd "$SLURM_SUBMIT_DIR"

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

srun -u wem-grid-hrrr \
  --era5-grid ./era5_grid.csv \
  --data-dir /datasets/WIND/HRRR
