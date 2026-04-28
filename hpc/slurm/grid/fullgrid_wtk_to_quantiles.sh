#!/bin/bash
#SBATCH --job-name=wtkq
#SBATCH --output=wtk_logs/wtkq_%j.out
#SBATCH --error=wtk_logs/wtkq_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --account=tap
#SBATCH --nodes=1

set -euo pipefail
mkdir -p wtk_logs
cd "$SLURM_SUBMIT_DIR"

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

srun -u wem-grid-wtk \
  --era5-grid ./era5_grid.csv \
  --wtk-dir /datasets/WIND/conus/v1.0.0/
