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
import time

class Prot_Hop:
    """MPI supporting postprocesses class for NVT VASP simulations of aqueous KOH."""

    def __init__(self, folder, T_ave=325, dt=0.5):
        """
        Postprocesses class for NVT VASP simulations of aqueous KOH.

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
        self.tstart = time.time()
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
        self.loop_timesteps_all2()
        # self.test_combining()
 
    def setting_properties_main(self, folder):
        """
        Load the output file into dedicated arrays
        
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
        """
        Initializes system for all cores
        
        This function initializes the variables that need to be available to all cores
        """
        
        # Import the correct split position arrays
        self.pos = self.comm.scatter(self.pos_all_split, root=0)
        self.L = self.comm.bcast(self.L, root=0)
        self.N = self.comm.bcast(self.N, root=0)
        
        self.n_max = len(self.pos[:, 0, 0])  # number of timestep on core
        if self.rank == 0:
            print('Rank', self.rank, "number timesteps", self.n_max)
            print('Time after communication', time.time() - self.tstart)
        
        # Asses the number of Hydrogens
        self.N_tot = np.sum(self.N)
        self.N_H = self.N[self.species.index('H')]
        self.N_O = self.N[self.species.index('O')]
        self.N_K = self.N[self.species.index('K')]
        
        self.H = np.arange(self.N_H)
        self.O = np.arange(self.N_H, self.N_H + self.N_O)
        self.K = np.arange(self.N_H + self.N_O, self.N_H + self.N_O + self.N_K)
        
        # Calculate and initiate the OH- tracking
        self.n_OH = self.N_K  # intended number of OH- from input file

    # def find_O_i(self, n):
    #     """
    #     Find the index of the Oxygen beloging to the OH-.

    #     This function searches for the index of the Oxygen belonging to the OH-
    #     particles. It automatically creates a neighbor list as well as an array
    #     which holds the unwraped location of the real OH particle.
        
    #     Args:
    #         n (integer): timestep of the assesment of the hydoxide recognition
    #     """
    #     if n == 0:  # Do a startup procedure
    #         self.OH = np.zeros(self.n_max, self.N_OH, 3)  # prepair the OH- position storage array
    #         self.OH_i = np.zeros(self.n_max, self.N_OH)  # prepair for the OH- O index storage array
    #         self.n_OH = np.zeros(self.n_max)  # prepair for real number of OH-
    #         self.OH_shift = np.zeros((self.n_OH, 1, 3))
    
    #         self.H2O = np.zeros(self.n_max, self.N_O - self.N_OH, 3)  # prepair the H2O position storage array
    #         self.H2O_i = np.zeros(self.n_max, self.N_OH)  # prepair for the H2O O index storage array
    #         self.n_H2O = np.zeros(self.n_max)  # prepair for real number of H2O
    #         self.H2O_shifts = np.zeros((self.N_O-self.n_OH, 3))
            

    def loop_timesteps_all1(self):
        # select only upper triangle interactions and create masks to be able to split species
        indices = np.triu_indices(self.N_tot, k=1)
        get_HH = np.where((np.isin(indices, self.H)[0]==True) & (np.isin(indices, self.H)[1]==True))[0]
        get_HO = np.where((np.isin(indices, self.H)[0]==True) & (np.isin(indices, self.O)[1]==True))[0]
        get_HK = np.where((np.isin(indices, self.H)[0]==True) & (np.isin(indices, self.K)[1]==True))[0]
        get_OO = np.where((np.isin(indices, self.O)[0]==True) & (np.isin(indices, self.O)[1]==True))[0]
        get_OK = np.where((np.isin(indices, self.O)[0]==True) & (np.isin(indices, self.K)[1]==True))[0]
        get_KK = np.where((np.isin(indices, self.K)[0]==True) & (np.isin(indices, self.K)[1]==True))[0]
        for i in range(self.n_max):
            # Calculate all interatomic distances with PBC and MIC
            pos = self.pos[i, :, :]
            r = (pos[indices[1] - indices[0]] + self.L/2) % self.L - self.L/2
            d = np.sqrt(np.sum(r**2, axis=1))
        if self.rank == 0:
            print('Time completion', time.time() - self.tstart)
    
    # def loop_timesteps_all2(self):     # DEPRICATED, is slower and paralizes less
    #     # select only upper triangle interactions and create masks to be able to split species
    #     for i in range(self.n_max):
    #         # Calculate all interatomic distances with PBC and MIC
    #         r = np.broadcast_to(self.pos[i, :, :], (self.N_tot, self.N_tot, 3))
    #         r_vect = (r - r.transpose(1, 0, 2) + self.L/2) % self.L - self.L/2
    #         d = np.sqrt(np.einsum('ijk, ijk->ij', r_vect, r_vect, optimize='optimal'))
    #     if self.rank == 0:
    #         print('Time completion', time.time() - self.tstart)  
        
    def test_combining(self):
        outputData = self.comm.gather(self.pos, root=0)
        if self.rank == 0:
            outputData = np.concatenate(outputData,axis = 0)
            print(np.array_equal(outputData, self.pos_all))
        print('Rank=', self.rank, 'result', self.n_OH)


Traj = Prot_Hop("../../../RPBE_Production/MLMD/100ps_Exp_Density/i_1", dt=0.5)

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