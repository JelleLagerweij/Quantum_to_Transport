# Post-processing pure water simulations

In this script, it is assumed that the pure water simulations are run for all 4 settings (AIMDxMLMD, RPBE-D3xrVV10 r2SCAN) and of each of these there are 5 different runs for statistics.

These output files should be placed in this folder, in sub directories with the name "'RPBE_AIMD', 'RPBE_MLMD', 'r2SCAN_AIMD', 'r2SCAN_MLMD'" and subsub directories ordered as "'i_1', 'i_2' , 'i_3', 'i_4', 'i_5'".
Reference RDFs are provided as well, those with "exp" in there are experimental: DOI 10.1088/0953-8984/19/33/335206. Those with rpbe in the name are calculated in: DOI 10.1063/1.4892400.