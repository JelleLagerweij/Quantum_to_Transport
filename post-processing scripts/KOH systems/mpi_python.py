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
import sys
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
        self.error_flag = 0

        # Normal Startup Behaviour
        self.setting_properties_all(folder)  # all cores
        self.loop_timesteps_all()
        self.stitching_together_all()
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

        # split array up
        self.L = data['lattice_vectors'][0, 0, 0]
        self.pos_all = self.L*data['positions']
        self.pos_all_split = np.array_split(self.pos_all, self.size, axis=0)
        
        # calculate chunk sizes for communication
        self.chunks = [int]*self.size
        self.steps_split = [None]*self.size
        sta = 0
        for i in range(self.size):
            self.chunks[i] = len(self.pos_all_split[i][:, 0])
            sto = sta + len(self.pos_all_split[i][:, 0])
            self.steps_split[i] = np.arange(start=sta, stop=sto)
            sta = sto + 1
        
        # self.steps = np.arange(self.pos_all.shape[0])
        # print(self.steps.size)
        # self.steps_split = np.array_split(self.pos_all, self.size)
        # print(self.steps.size)
        # Todo lateron:
        # 1. Load the stresses
        # 2. Load the energies

    def setting_properties_all(self, folder):
        """
        Initializes system for all cores

        This function initializes the variables that need to be available to all cores
        """
        if self.rank == 0:
            self.setting_properties_main(folder)  # Initializes on main cores
        else:
            self.chunks = [None]*self.size
            self.L = float
            self.N = int
        
        self.chunks = self.comm.bcast(self.chunks, root=0)
        self.L = self.comm.bcast(self.L, root=0)
        self.N = self.comm.bcast(self.N, root=0)
        
        # Asses the number of Hydrogens
        self.N_tot = np.sum(self.N)
        self.N_H = self.N[self.species.index('H')]
        self.N_O = self.N[self.species.index('O')]
        self.N_K = self.N[self.species.index('K')]

        self.H = np.arange(self.N_H)
        self.O = np.arange(self.N_H, self.N_H + self.N_O)
        self.K = np.arange(self.N_H + self.N_O, self.N_H + self.N_O + self.N_K)

        # Calculate and initiate the OH- tracking
        self.N_OH = self.N_K  # intended number of OH- from input file
        self.N_H2O = self.N_O - self.N_OH
        
        if self.rank != 0:
            self.pos_all_split = [np.empty((self.chunks[i], self.N_tot, 3)) for i in range(self.size)]  # create empty dumy on all cores
            self.steps_split = [np.empty(self.chunks[i]) for i in range(self.size)]

        # Import the correct split position arrays
        self.pos = self.comm.scatter(self.pos_all_split, root=0)
        self.steps = self.comm.scatter(self.steps_split, root=0)

        self.n_max = len(self.pos[:, 0, 0])  # number of timestep on core
        if self.rank == 0:
            # print('Rank', self.rank, "number timesteps", self.n_max)
            print('Time after communication', time.time() - self.tstart)


    def recognize_molecules(self, n):
        """
        Find the index of the Oxygen beloging to the OH- or H2O.

        This function searches for the index of the Oxygen belonging to the OH-
        particles. It automatically creates a neighbor list as well as an array
        which holds the unwraped location of the real OH particle.

        Args:
            n (integer): timestep of the assesment of the hydoxide recognition
        """
        if n == 0:  # Do a startup procedure
            self.OH = np.zeros((self.n_max, self.N_OH, 3))  # prepair the OH- position storage array
            self.OH_i = np.zeros((self.n_max, self.N_OH), dtype='int')  # prepair for the OH- O index storage array
            self.n_OH = np.zeros(self.n_max, dtype='int')  # prepair for real number of OH-
            self.OH_shift = np.zeros((self.N_OH, 3), dtype='int')  # prepair history shift list for pbc crossings

            self.H2O = np.zeros((self.n_max, self.N_H2O, 3))  # prepair the H2O position storage array
            self.H2O_i = np.zeros((self.n_max, self.N_H2O), dtype='int')  # prepair for the H2O O index storage array
            self.n_H2O = np.zeros(self.n_max, dtype='int')  # prepair for real number of H2O
            self.H2O_shift = np.zeros((self.N_O-self.N_OH, 3), dtype='int')  # prepair history shift list for pbc crossings
            
            self.n_H3O = np.zeros(self.n_max, dtype='int')  # prepair for real number of H3O+ (should be 0)
        
        counter_H_per_O = np.bincount(np.argmin(self.d_HO.reshape((self.N_H, self.N_O)), axis=1))  # Get number of H per oxygen in molecule
        
        # Identify and count all real OH-, H2O and H3O+
        OH_i = np.where(counter_H_per_O == 1)[0]
        self.n_OH[n] = OH_i.shape[0]
        H2O_i = np.where(counter_H_per_O == 2)[0]
        self.n_H2O[n] = H2O_i.shape[0]
        H3O_i = np.where(counter_H_per_O == 3)[0]
        self.n_H3O[n] = H3O_i.shape[0]
        
        # Now start matching the correct molecules together
        if n == 0:
            # No matching needed, just filling the arrays correctly
            self.OH_i[n, :] = OH_i
            self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :]
            
            
            self.H2O_i[n, :] = H2O_i
            self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :]
            
            self.OH_i_s = OH_i  # set this in the first timestep
        else:
            if (OH_i == self.OH_i_s).all():  # No reaction occured only check PBC
                self.OH_i[n, :] = self.OH_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                
                self.H2O_i[n, :] = self.H2O_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :] + self.L*self.H2O_shift

            else:  # Normal reaction cases
                # find which OH- belongs to which OH-. This is difficult because of sorting differences.
                
                if self.N_OH != self.n_OH[n] or self.n_H3O[n] > 0:  # Exemption for H3O+ cases
                    print(f"Strange behaviour found, H3O+ created. Rank", self.rank, 'timestep ', self.steps[n])

                diff_new = np.setdiff1d(OH_i, self.OH_i[n-1, :], assume_unique=True)
                diff_old = np.setdiff1d(self.OH_i[n-1, :], OH_i, assume_unique=True)
                
                self.OH_i[n, :] = self.OH_i[n-1, :]
                self.H2O_i[n, :] = self.H2O_i[n-1, :]
                for i in range(len(diff_old)):
                    ## HYDROXIDE PART
                    # Check every closest old version to every unmatching new one and replace the correct OH_i index
                    r_OO = (self.pos_O[n, diff_new, :] - self.pos_O[n, diff_old[i], :] + self.L/2) % self.L - self.L/2               
                    d2 = np.sum(r_OO**2, axis=1)
                    i_n = np.argmin(d2)  # find new OH- index

                    if d2[i_n] > 9:  # excape when jump too large
                        print("Strange behaviour found, reaction jump too far, d = ", np.sqrt(d2[i_n]), 'Angstrom',
                                        "CPU rank=", self.rank, 'timestep', self.steps[n])

                    idx_OH_i = np.where(self.OH_i[n-1, :] == diff_old[i]) # get index in OH_i storage list
                    self.OH_i[n, idx_OH_i] = diff_new[i_n]  # update OH_i storage list

                    # Adjust for different PBC shifts in reaction
                    dis = (self.pos_O[n, self.OH_i[n, idx_OH_i], :] - self.pos_O[n-1, self.OH_i[n-1, idx_OH_i], :] + self.L/2) % self.L - self.L/2   # displacement vs old location
                    real_loc = self.pos_O[n-1, self.OH_i[n-1, idx_OH_i], :] + dis
                    self.OH_shift[idx_OH_i, :] += np.round((real_loc - self.pos_O[n, self.OH_i[n, idx_OH_i], :])/self.L).astype(int)  # Correct shifting update for index
                
                    ## WATER PART
                    # We already know the exchange of atomic indexes, now find the array location in the water list (logic is reversed)
                    idx_H2O_i = np.where(self.H2O_i[n-1, :] == diff_new[i_n])  # New OH- was old H2O
                    self.H2O_i[n, idx_H2O_i] = diff_old[i]  # Therefore new H2O is old OH-
                    
                    # Adjust for different PBC shifts in reaction
                    dis = (self.pos_O[n, self.H2O_i[n, idx_H2O_i], :] - self.pos_O[n-1, self.H2O_i[n-1, idx_H2O_i], :] + self.L/2) % self.L - self.L/2   # displacement vs old location
                    real_loc =self.pos_O[n-1, self.H2O_i[n-1, idx_H2O_i], :] + dis
                    self.H2O_shift[idx_H2O_i, :] += np.round(real_loc - self.pos_O[n, self.H2O_i[n, idx_H2O_i], :]).astype(int)  # Correct shifting update for index
                
                # Update all the positions
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :]+ self.L*self.H2O_shift
                self.OH_i_s = OH_i  # always sort after reaction or initiation to have a cheap check lateron.
   
    def loop_timesteps_all(self, n_samples=10, cheap=True): 
        """This function loops over all timesteps and tracks all over time properties
        
        The function tracks calls the molecule recognition function and the rdf functions when needed.

        Args:
            n_samples (int, optional): time between sampling rdfs. Defaults to 10.
            cheap (bool, optional): skips less relevant interaction modes. Defaults to True.
        """
        # split the arrays up to per species description
        self.pos_H = self.pos[:, self.H, :]
        self.pos_O = self.pos[:, self.O, :]
        self.pos_K = self.pos[:, self.K, :]

        # Create per species-species interactions indexing arrays
        idx_HO = np.mgrid[0:self.N_H, 0:self.N_O].reshape(2, self.N_H*self.N_O)
        idx_OO = np.triu_indices(self.N_O, k=1)
        idx_KO = np.mgrid[0:self.N_K, 0:self.N_O].reshape(2, self.N_K*self.N_O)
        idx_KK = np.triu_indices(self.N_K, k=1)
        if cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
            idx_HH = np.triu_indices(self.N_H, k=1)
            idx_HK = np.mgrid[0:self.N_H, 0:self.N_K].reshape(2, self.N_H*self.N_K)

        for n in range(self.n_max):  # Loop over all timesteps
            # Calculate only OH distances for OH- recognition
            r_HO = (self.pos_O[n, idx_HO[1], :] - self.pos_H[n, idx_HO[0], :] + self.L/2) % self.L - self.L/2
            self.d_HO = np.sqrt(np.sum(r_HO**2, axis=1))
            
            self.recognize_molecules(n)
            if n % n_samples == 0:
                # Calculate all other distances for RDF's and such when needed
                r_OO = (self.pos_O[n, idx_OO[1], :] - self.pos_O[n, idx_OO[0], :] + self.L/2) % self.L - self.L/2
                self.d_OO = np.sqrt(np.sum(r_OO**2, axis=1))
                
                r_KO = (self.pos_O[n, idx_KO[1], :] - self.pos_K[n, idx_KO[0], :] + self.L/2) % self.L - self.L/2
                self.d_KO = np.sqrt(np.sum(r_KO**2, axis=1))
                
                r_KK = (self.pos_K[n, idx_KK[1], :] - self.pos_K[n, idx_KK[0], :] + self.L/2) % self.L - self.L/2
                self.d_KK = np.sqrt(np.sum(r_KK**2, axis=1))

                if cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
                    r_HH = (self.pos_H[n, idx_HH[1], :] - self.pos_H[n, idx_HH[0], :] + self.L/2) % self.L - self.L/2
                    self.d_HH = np.sqrt(np.sum(r_HH**2, axis=1))
                    
                    r_HK = (self.pos_K[n, idx_HK[1], :] - self.pos_H[n, idx_HK[0], :] + self.L/2) % self.L - self.L/2
                    self.d_HK = np.sqrt(np.sum(r_HK**2, axis=1))
            
            

        if self.rank == 0:
            print('Time calculating distances', time.time() - self.tstart)

    def stitching_together_all(self):
        # prepair gethering on all cores 1) All OH- stuff
        self.OH_i = self.comm.gather(self.OH_i, root=0)
        self.OH = self.comm.gather(self.OH, root=0)
        self.n_OH = self.comm.gather(self.n_OH, root=0)
        self.OH_shift = self.comm.gather(self.OH_shift, root=0)
        
        # prepair gethering on all cores 1) All H2O stuff
        self.H2O_i = self.comm.gather(self.H2O_i, root=0)
        self.H2O = self.comm.gather(self.H2O, root=0)
        self.n_H2O = self.comm.gather(self.n_H2O, root=0)
        self.H2O_shift = self.comm.gather(self.H2O_shift, root=0)
        
        # prepair gethering on all cores 1) All H3O+ stuff
        self.n_H3O = self.comm.gather(self.n_H3O, root=0)
        
        # RDF's
        # TODO
        
        # Stich together correctly on main cores
        if self.rank == 0:
            self.stitching_together_main()

        
    def stitching_together_main(self):
        # First handle arrays which do not need special conciderations
        
        self.n_H2O = np.concatenate(self.n_H2O, axis=0)
        self.n_H3O = np.concatenate(self.n_H3O, axis=0)
        
        # NOW ADD 1 reaction recognition and 2 reordering
        # for every end of 1 section check with start next one
        for n in range(self.size-1):
            mismatch_indices = np.where(self.OH_i[n][-1, :] != self.OH_i[n+1][0, :])[0]

            # Swap columns in C_modified based on the mismatch
            for index in mismatch_indices:
                # Find the index in B corresponding to the value in A
                b_index = np.where(self.OH_i[n+1][0, :] == self.OH_i[n][-1, index])[0]
            
                if b_index.size == 0:
                    print('skip, implement reaction here')
                elif b_index[0] == index:
                    print('nothing to do')
                else:
                    print('no skip')
                    # Get the range of columns to swap based on the difference in mismatched indices
                    # Swap the columns of OH index, OH position and the OH_shift arrays
                    self.OH_i[n+1][:, [index, b_index[0]]] = self.OH_i[n+1][:, [b_index[0], index]]
                    self.OH[n+1][:, [index, b_index[0]], :] = self.OH[n+1][:, [b_index[0], index], :]
                    self.OH_shift[n+1][[index, b_index[0]], :] = self.OH_shift[n+1][[b_index[0], index], :]
            
            # Now adjust for passing PBC in the shift collumns.
            self.OH[n+1][:, :, :] += self.L*self.OH_shift[n][:, :]
            self.OH_shift[n+1][:, :] += self.OH_shift[n][:, :]
        
        # Combining adjusted arrays back to the right shape
        self.OH = np.concatenate(self.OH, axis=0)
        self.OH_i = np.concatenate(self.OH_i, axis=0)
        self.n_OH = np.concatenate(self.n_OH, axis=0)
        
                
    def test_combining(self):
        outputData = self.comm.gather(self.pos, root=0)
        if self.rank == 0:
            outputData = np.concatenate(outputData,axis = 0)
            print('time to completion',  time.time() - self.tstart)
            
            plt.figure()
            # plt.plot(self.OH[:, 0, :], label=['x', 'y', 'z'])
            plt.plot(self.OH_i, label='OH- index ')
            for i in range(1, self.size):
                plt.axvline(x = self.pos_O.shape[0]*i, color = 'c')
            
            plt.xlim(0, self.size*self.pos_O.shape[0])
            plt.legend()
            
            plt.savefig(r'C:\Users\vlagerweij\Documents\TU jaar 6\Project KOH(aq)\Repros\Quantum_to_Transport\post-processing scripts\KOH systems\index_OH_mpi.png')

# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/", dt=0.5)
# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1", dt=0.5)
Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1", dt=0.5)
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/", dt=0.5)