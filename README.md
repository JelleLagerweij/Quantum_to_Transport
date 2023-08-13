# Quantum_to_Transport
## Modeling Transport Properties of Aqueous Potassium Hydroxide by Machine Learning Molecular Force Fields from Quentum Mechanics

In this repository all relavant simulation settings, input files, execute commands and post-processing scrips for the master thesis with the same title are availlable.
All simulations are performed with VASP 6.4.1, compiled with gcc 11. Modules loaded contain openmpi, openblas, netlib-scalapack, fftw and hdf5.

## KOH density
The simulations prepared for checking the equilibrium density of KOH are contained in this folder.
Note that only 1 POSCAR is provided, and the box size has to be manually changed, by adjusting the first 4 lines of the POSCAR.
The "bozsizes.py" python code provides the right lengths to input given a certain density.


## KOH production


## KOH validation


## post-processing scripts
In the post-processing scripts folder all post-prossesing scripts of the pure water, the KOH validation and the KOH production simulations.

## Pure water
In the subdirectory with this name, the simulation settings and input files of a system of 64 waters molecule is availlable.