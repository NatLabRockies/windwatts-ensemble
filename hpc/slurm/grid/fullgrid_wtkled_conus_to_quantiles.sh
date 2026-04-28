#!/bin/bash
#SBATCH --job-name=wtkq
#SBATCH --output=wtkled_logs/wtkledq_%j.out
#SBATCH --error=wtkled_logs/wtkledq_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --partition=long
#SBATCH --account=cscdav
#SBATCH --nodes=1

set -euo pipefail
mkdir -p wtkled_logs

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

srun -u wem-grid-wtkled \
      	--era5-grid ./era5_grid.csv \
      	--data-dir /datasets/WIND/conus/v2.0.0 \
      	--out-dir out_wtkled \
      	--format csv
