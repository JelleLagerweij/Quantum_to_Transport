# Quantum_to_Transport
## Modeling Transport Properties of Aqueous Potassium Hydroxide by Machine Learning Molecular Force Fields from Quentum Mechanics

In this repository all relavant simulation settings, input files, execute commands and post-processing scrips for the master thesis with the same title are availlable.
All simulations are performed with VASP 6.4.1, compiled with gcc 11. Modules loaded contain openmpi, openblas, netlib-scalapack, fftw and hdf5.

## KOH density
The simulations prepared for checking the equilibrium density of KOH are contained in this folder.
Note that only 1 POSCAR is provided, and the box size has to be manually changed, by adjusting the first 4 lines of the POSCAR.
The "bozsizes.py" python code provides the right lengths to input given a certain density. Note that the "runVASP_GCC" file is prepaired for a slurm scheduler, however the user needs to adjust the path to their VASP 6.4.1 instalation.
These simulations use 5 ps of equilibriation after which the results are coppied to a initation folder and the 20ps simulations are started.

## KOH production
The KOH production calculations are used to compute the main results of this work. These are prepaired to run immidiately if intended, starting up 5 different calculations with different atomic starting postions and velocities.
The simulations can be ran using the "run.sh" script. In this script the link to the POSCARS can be changed to be either at 1335kg/m^3 or 1048kg/m^3, the calculated and experimental equilibrium density.
Do not run as is, as the POTCAR is not provided due to it being interlectual properties of VASP.

## KOH validation
The KOH validation simulations are similar to KOH production. They are shorter: only 10 ps, and AIMD and MLMD runs can both be executed.

## post-processing scripts
In the post-processing scripts folder all post-prossesing scripts of the pure water, the KOH validation and the KOH production simulations.
Additional instructions are provided in the subfolders of the post-processing scripts.

## Pure water
In the subdirectory with this name, the simulation settings and input files of a system of 64 waters molecule is availlable.
Four different INCARS are provided, mixing AIMD and MLMD as well as RPPBE-D3 and rVV10 r2SCAN.

This is the old version, which is slow compared to the improved implementation