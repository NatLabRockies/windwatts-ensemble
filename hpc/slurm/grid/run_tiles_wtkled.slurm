#!/bin/bash
#SBATCH --job-name=wtkled_tiles
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00        
#SBATCH --nodes=1
#SBATCH --account=cscdav

mkdir -p logs

ml conda
conda activate "${WEM_CONDA_ENV:-/scratch/kmenear/windwatts/env}"

# Run your bash driver
bash run_tiles_locally_wtkled.sh splits_wtkled_v2/missing_tiles_wtkled_part_02.txt
