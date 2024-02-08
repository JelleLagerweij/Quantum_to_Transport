import numpy as np
import pandas as pd
import scipy as sp
import re
import scipy.constants as co
import scipy.optimize as opt
import matplotlib.pyplot as plt
import freud
from py4vasp import Calculation
from mpi4py import MPI

class Prot_Hop:
    """MPI supporting postprocesses class for NVT VASP simulations of aqueous KOH."""
    
    def __init__(self, folder, T_ave=325, dt=0.5e-15):
        """Postprocesses class for NVT VASP simulations of aqueous KOH.

        It can compute the total system viscosity and the diffusion coefficient
        of the K+ and the OH-. The viscosity is computed using Green-Kubo from
        the pressure correlations and the diffusion coefficient using Einstein
        relations from the positions. The reactive OH- trajactory is computed
        using minimal and maximal bond lengths.

        Args:
            folder (srting): path to hdf5 otput file
            T_ave (float/int, optional): The set simulation temperature. Defaults to 325.
            dt (float/int, optional): The simulation timestep. Defaults to 0.5e-15.
        """
        self.dt = dt
        self.species = ['H', 'O', 'K']
        self.T_ave = T_ave
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Normal Startup Behaviour
        # Initialize system on the main core
        if rank == 0:
            self.setting_properties(folder)
        # Transport all chunks to relavant arrays
        a = 0
        
    def setting_properties(self, folder):
        """Load the output file into dedicated arrays
        
        This function should only be executed on the main core

        Args:
            folder (string): path to hdf5 otput file
        """
        
            

# Initialize and load the entire system
   

# if rank == 0:
#     test = np.random.rand(2, 4, 3)  # i_timesteps, j_atoms, k_dim
#     print("Test Array", test)
#     test_chunks = np.array_split(test, size, axis=0) # split it in time blocks
# else:
#     test_chunks = None

# test_chunk = comm.scatter(test_chunks, root=0)  # send the blocks to all cores
# output_chunk = transformation1(test_chunk)
# print("1 Chunked Array numero", rank, output_chunk)

# output_chunk = transformation2(output_chunk)
# print("2 Chunked Array numero", rank, output_chunk)
# outputData = comm.gather(output_chunk, root=0)
# if rank == 0:
#     outputData = np.concatenate(outputData,axis = 0)
#     print("Output Array", outputData)
