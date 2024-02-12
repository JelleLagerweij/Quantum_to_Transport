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
        self.loop_timesteps_all()
        self.test_combining()

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

    def find_O_i(self, n):
        """
        Find the index of the Oxygen beloging to the OH-.

        This function searches for the index of the Oxygen belonging to the OH-
        particles. It automatically creates a neighbor list as well as an array
        which holds the unwraped location of the real OH particle.

        Args:
            n (integer): timestep of the assesment of the hydoxide recognition
        """
        if n == 0:  # Do a startup procedure
            self.OH = np.zeros(self.n_max, self.N_OH, 3)  # prepair the OH- position storage array
            self.OH_i = np.zeros(self.n_max, self.N_OH)  # prepair for the OH- O index storage array
            self.n_OH = np.zeros(self.n_max)  # prepair for real number of OH-
            self.OH_shift = np.zeros((self.n_OH, 1, 3))  # prepair history shift list for pbc crossings

            self.H2O = np.zeros(self.n_max, self.N_O - self.N_OH, 3)  # prepair the H2O position storage array
            self.H2O_i = np.zeros(self.n_max, self.N_OH)  # prepair for the H2O O index storage array
            self.n_H2O = np.zeros(self.n_max)  # prepair for real number of H2O
            self.H2O_shifts = np.zeros((self.N_O-self.n_OH, 3))  # prepair history shift list for pbc crossings
        

    def loop_timesteps_all(self, n_samples=10, cheap=True):     # DEPRICATED, is slower and paralizes less
        # split the arrays up to per species description
        pos_H = self.pos[:, self.H, :]
        pos_O = self.pos[:, self.O, :]
        pos_K = self.pos[:, self.K, :]

        # Create per species-species interactions indexing arrays
        idx_HO = np.mgrid[0:self.N_H, 0:self.N_O].reshape(2, self.N_H*self.N_O)
        idx_OO = np.triu_indices(self.N_O, k=1)
        idx_KO = np.mgrid[0:self.N_K, 0:self.N_O].reshape(2, self.N_K*self.N_O)
        idx_KK = np.triu_indices(self.N_K, k=1)
        if cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
            idx_HH = np.triu_indices(self.N_H, k=1)
            idx_HK = np.mgrid[0:self.N_H, 0:self.N_K].reshape(2, self.N_H*self.N_K)

        for i in range(self.n_max):
            # Calculate only OH distances for OH- recognition
            r_HO = (pos_O[i, idx_HO[1], :] - pos_H[i, idx_HO[0], :] + self.L/2) % self.L - self.L/2
            d_HO = np.sqrt(np.sum(r_HO**2, axis=1))
            
            self.find_O_i(i)
            if i % n_samples == 0:
                # Calculate all other distances for RDF's and such when needed
                r_OO = (pos_O[i, idx_OO[1], :] - pos_O[i, idx_OO[0], :] + self.L/2) % self.L - self.L/2
                d_OO = np.sqrt(np.sum(r_OO**2, axis=1))
                r_KO = (pos_O[i, idx_KO[1], :] - pos_K[i, idx_KO[0], :] + self.L/2) % self.L - self.L/2
                d_KO = np.sqrt(np.sum(r_KO**2, axis=1))
                r_KK = (pos_K[i, idx_KK[1], :] - pos_K[i, idx_KK[0], :] + self.L/2) % self.L - self.L/2
                d_KK = np.sqrt(np.sum(r_KK**2, axis=1))

                if cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
                    r_HH = (pos_H[i, idx_HH[1], :] - pos_H[:, idx_HH[0], :] + self.L/2) % self.L - self.L/2
                    d_HH = np.sqrt(np.sum(r_HH**2, axis=1))
                    r_HK = (pos_K[i, idx_HK[1], :] - pos_H[:, idx_HK[0], :] + self.L/2) % self.L - self.L/2
                    d_HK = np.sqrt(np.sum(r_HK**2, axis=1))
            
            

        if self.rank == 0:
            print('Time calculating distances', time.time() - self.tstart)

    def test_combining(self):
        outputData = self.comm.gather(self.pos, root=0)
        if self.rank == 0:
            outputData = np.concatenate(outputData,axis = 0)
            print('Combining again is', np.array_equal(outputData, self.pos_all))
            print('time to completion',  time.time() - self.tstart)


# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/", dt=0.5)
# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1", dt=0.5)
Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1", dt=0.5)
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/", dt=0.5)
