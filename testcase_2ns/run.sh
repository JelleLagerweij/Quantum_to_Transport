#!/bin/bash

for i in {1..5}
do
    cp -r run_0 i_$i
    cp ../../AIMD/i_$i/CONTCAR i_$i/POSCAR
    cd i_$i
    sed -i 's/JOBID/MLFF production run '$i'/g' runVASP
    sbatch runVASP
    cd ..
done