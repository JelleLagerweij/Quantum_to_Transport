#!/bin/bash
#
#SBATCH --job-name='Test Corruption'
#SBATCH --partition=compute-p2
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=research-ME-pe
#SBATCH --mem=16G

# Load modules:
module load 2023r1-gcc11
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate vasp_post
srun python testcase.py
conda deactivate

seff $SLURM_JOBID