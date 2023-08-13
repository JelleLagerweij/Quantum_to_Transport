#!/bin/bash
runfile=$(expr runVASP)  # Server where to run

mkdir 10ps
cd 10ps

for i in 1 2 3 4 5
do
	mkdir i_$i
	cd i_$i

	cp ../../0_input/$runfile runVASP
	cp ../../0_input/INCAR INCAR
	cp ../../0_input/POTCAR .
	cp ../../0_input/KPOINTS .
	cp ../../0_input/POSCARS/i_$i/POSCAR .

	sed -i 's/JOB_NAME/MLMD10ps_run_'$i'/' runVASP
	sbatch runVASP
	cd ..
done
cd ..
