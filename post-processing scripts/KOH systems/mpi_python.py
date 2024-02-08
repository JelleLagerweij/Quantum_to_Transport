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
    
    def __init__(self, folder, T_ave=325, dt=0.5):
        """Postprocesses class for NVT VASP simulations of aqueous KOH.

        It can compute the total system viscosity and the diffusion coefficient
        of the K+ and the OH-. The viscosity is computed using Green-Kubo from
        the pressure correlations and the diffusion coefficient using Einstein
        relations from the positions. The reactive OH- trajactory is computed
        using minimal and maximal bond lengths.

        Args:
            folder (string): path to hdf5 otput file
            T_ave (float/int, optional): The set simulation temperature in K. Defaults to 325.
            dt (float/int, optional): The simulation timestep in fs. Defaults to 0.5.
        """
        self.dt = dt
        self.species = ['H', 'O', 'K']
        self.T_ave = T_ave
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # Normal Startup Behaviour
        # Initialize system on the main core
        if self.rank == 0:
            self.setting_properties_main(folder)
        else:
            self.pos_all_split = None  # create empty dumy on all cores
            self.L = None
            self.N = None     

        # Transport all chunks and properties to relavant arrays
        self.setting_properties_all()
        self.test_combining()
        
    def setting_properties_main(self, folder):
        """Load the output file into dedicated arrays
        
        This function should only be executed on the main core

        Args:
            folder (string): path to hdf5 otput file
        """
        
        # import the system from the hdf5 output
        self.df = Calculation.from_path(folder)
        data = self.df.structure[:].to_dict()
        
        # Calculate number of atoms total and per type and their indexes
        self.N = np.array([data['elements'].count(self.species[0]),
                           data['elements'].count(self.species[1]),
                           data['elements'].count(self.species[2])])
        


        # Calculate the number of intended OH- ions
        self.L = data['lattice_vectors'][0, 0, 0]
        self.pos_all = self.L*data['positions']
        self.pos_all_split = np.array_split(self.pos_all, self.size, axis=0)
        
        # Todo lateron:
        # 1. Load the stresses
        # 2. Load the energies

    def setting_properties_all(self):
        """Initializes system for all cores
        """
        
        # Import the correct split position arrays
        self.pos = self.comm.scatter(self.pos_all_split, root=0)
        self.L = self.comm.bcast(self.L, root=0)
        self.N = self.comm.bcast(self.N, root=0)
        
        self.N_H = self.N[self.species.index('H')]
        self.N_O = self.N[self.species.index('O')]
        self.N_K = self.N[self.species.index('K')]
        
        self.H = np.arange(self.N_H)
        self.O = np.arange(self.N_H, self.N_H + self.N_O)
        self.K = np.arange(self.N_H + self.N_O, self.N_H + self.N_O + self.N_K)
        
        # Calculate and initiate the OH- tracking
        self.n_OH = 2*self.N_O- self.N_H  # intended number of OH- from input file
        self.shift = np.zeros((self.n_OH, 1, 3))
        # Set your shift to 0 for all OH- and give them random index
        # Find your first selection of OH-
    
    def test_combining(self):
        outputData = self.comm.gather(self.pos, root=0)
        if self.rank == 0:
            outputData = np.concatenate(outputData,axis = 0)
            print(np.array_equal(outputData, self.pos_all))
        print('Rank=', self.rank, 'result', self.n_OH)


Traj = Prot_Hop("../../../RPBE_Production/AIMD/10ps/i_1", dt=0.5)

# outputData = Prot_Hop.comm.gather(Prot_Hop.pos, root=0)
# if Prot_Hop.rank == 0:
#     print(outputData == Prot_Hop.pos_all)

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