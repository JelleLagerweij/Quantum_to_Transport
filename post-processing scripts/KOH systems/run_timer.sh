#!/usr/bin/env bash
#!/bin/bash

#SBATCH --job-name='test_mpi'
#SBATCH --partition=compute-p2
#SBATCH --time=0-00:30:00
#SBATCH --account=research-ME-pe

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --hint=multithread

module load 2023r1-gcc11
module load openmpi

module load miniconda3
conda activate vasp_post

echo "16 HT START"
mpiexec -n 16 --use-hwthread-cpus python mpi_python.py
echo "16 HT COMPLETED"

echo "8 START"
mpiexec -n 8 python mpi_python.py
echo "8 COMPLETED"

echo "4 START"
mpiexec -n 4 python mpi_python.py
echo "4 COMPLETED"

echo "2 START"
mpiexec -n 2 python mpi_python.py
echo "2 COMPLETED"

echo "1 MPI START"
mpiexec -n 1 python mpi_python.py
echo "1 MPI COMPLETED"

echo "1START"
python mpi_python.py
echo "1 COMPLETED"