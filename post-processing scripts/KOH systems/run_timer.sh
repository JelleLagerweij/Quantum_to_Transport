#!/usr/bin/env bash


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