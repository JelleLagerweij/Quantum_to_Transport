# Post-Processing KOH (aq) systems
## Classfile
The "Class_diff_hopping_hdf5.py" code holds the object oriented python class file. In this file, methods to extract the system properties out of the HDF5 output are contained.
This class contains methods to track the OH-, K+ and H2O particles as well as how to get their MSDs and diffusion coefficients. Additionally, in can compute viscosity from stress tensor autocorrelations (Green-Kubo) as well as methods to retrieve system energies, temperature, pressure and ML error estimations.

This class is used by executing the "compute.py" python code. As the VASP output files are large in size (unsuitable for Github) only a single 10ps MLMD result is provided. This is too short to retrieve trustworthy transport properties from, however it showcases the workings of the code.
Note that if VASP is compiled with HDF5, all relavant properties are stored in the vaspout.h5 file, making data handling much more practical. The newly added ML is not implemented yet in the HDF5 format of VASP. The following command can extract the correct error data out of the ML_LOGFILE:
```
cat ML_LOGFILE | grep BEEF > BEEF.dat
```