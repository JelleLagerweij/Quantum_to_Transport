import numpy as np
import glob
import matplotlib.pyplot as plt
import freud
# from py4vasp import Calculation
from mpi4py import MPI
import time
import os
import h5py
import ase
import sys
from ase import io

class Prot_Hop:
    """MPI supporting postprocesses class for NVT VASP simulations of aqueous KOH."""

    def __init__(self, folder, T_ave=325, cheap=True, verbose=False, xyz_out=False, serial_check=False):
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
        """
        self.tstart = time.time()
        self.species = ['H', 'O', 'K']

        self.comm = MPI.COMM_WORLD
        
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.error_flag = 0
        self.cheap = cheap
        self.folder = os.path.normpath(folder)
        self.verbose = verbose
        self.xyz_out = xyz_out
        self.serial_check = serial_check

        # Normal Startup Behaviour
        self.setting_properties_all()  # all cores
        self.loop_timesteps_all()
        self.stitching_together_all()
        
        # afterwards on single core
        if self.rank == 0:
            self.compute_MSD()
            
        self.save_results_all()

        if self.rank == 0:
            if self.verbose is True:
                print('time to completion',  time.time() - self.tstart)

            if serial_check is True:
                self.test_combining_main()
        exit()
        
    def setting_properties_main(self):
        """
        Load the output file into dedicated arrays

        This function should only be executed on the main core

        Args:
            folder (string): path to hdf5 otput file
        """

        # finds all vaspout*.h5 files in folder and orders them alphabetically
        subsimulations = sorted(glob.glob(self.folder+r"/vaspout*"))
        if len(subsimulations) == 0:
            print(f'Fatal error, no such vaspout*.h5 file availlable at {self.folder}')
            self.comm.Abort()

        # loop over subsimulations
        for i in range(len(subsimulations)):
            self.df = h5py.File(subsimulations[i])
            if i == 0:
                try:
                    skips =  self.df['/input/incar/ML_OUTBLOCK'][()]
                except:
                    skips = 1
                self.pos_all = self.df['intermediate/ion_dynamics/position_ions'][()]
                self.force = self.df['intermediate/ion_dynamics/forces'][()]
                self.stress = self.df['intermediate/ion_dynamics/stress'][()]
                self.energy = self.df['intermediate/ion_dynamics/energies'][()]
                
                
                # Load initial structure properties and arrays out of first simulation
                self.N = self.df['results/positions/number_ion_types'][()]
                self.L = self.df['results/positions/lattice_vectors'][()][0, 0]  # Boxsize
                self.pos_all *= self.L
                
                # load initial force properties and arrays out of first simulation                
                # Read custom data out of HDF5 file (I do not know how to get it out of py4vasp)
                self.dt = self.df['/input/incar/POTIM'][()]
                self.dt *= skips
                self.T_set = float(self.df['/input/incar/TEBEG'][()])

            else:
                pos = self.df['intermediate/ion_dynamics/position_ions'][()]*self.L
                force = self.df['intermediate/ion_dynamics/forces'][()]
                stress = self.df['intermediate/ion_dynamics/stress'][()]
                energy = self.df['intermediate/ion_dynamics/energies'][()]
                               
                # load new positions, but apply pbc unwrapping (# ARGHGH VASP)
                dis_data = self.pos_all[-1, :, :] - pos[0, :, :]
                dis_real = ((self.pos_all[-1, :, :] - pos[0, :, :] + self.L/2) % self.L - self.L/2)
                pos -= (dis_real - dis_data)

                # now matching together
                self.pos_all = np.concatenate((self.pos_all, pos), axis=0)
                
                # load new forces, stresses and energies and add to old array
                self.force = np.concatenate((self.force, force), axis=0)
                self.stress = np.concatenate((self.stress, stress), axis=0)
                self.energy = np.concatenate((self.energy, energy), axis=0)
            self.df.close()

        self.t = np.arange(self.pos_all.shape[0])*self.dt
        
        # After putting multiple simulations together
        self.pos_all_split = np.array_split(self.pos_all, self.size, axis=0)
        self.force_split = np.array_split(self.force, self.size, axis=0)
        self.t_split = np.array_split(self.t, self.size)

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

    def setting_properties_all(self):
        """
        Initializes system for all cores

        This function initializes the variables that need to be available to all cores
        """
        if self.rank == 0:
            self.setting_properties_main()  # Initializes on main cores
        else:
            self.chunks = [None]*self.size
            self.L = float
            self.N = np.empty(3, dtype=int)
            self.dt = float
            self.T_set = float
        
        self.chunks = self.comm.bcast(self.chunks, root=0)
        self.L = self.comm.bcast(self.L, root=0)
        self.N = self.comm.bcast(self.N, root=0)
        self.dt = self.comm.bcast(self.dt, root=0)
        self.T_set =  self.comm.bcast(self.T_set, root=0)
        
        # print('self.dt is', self.dt, 'self.T_set is', self.T_set, 'rank is', self.rank)
        # Asses the number of Hydrogens
        self.N_tot = np.sum(self.N)
        self.N_H = self.N[self.species.index('H')]
        self.N_O = self.N[self.species.index('O')]
        self.N_K = self.N[self.species.index('K')]

        self.H_i = np.arange(self.N_H)
        self.O_i  = np.arange(self.N_H, self.N_H + self.N_O)
        self.K_i = np.arange(self.N_H + self.N_O, self.N_H + self.N_O + self.N_K)

        # Calculate and initiate the OH- tracking
        self.N_OH = self.N_K  # intended number of OH- from input file
        self.N_H2O = self.N_O - self.N_OH
        
        if self.rank != 0:
            self.pos_all_split = [np.empty((self.chunks[i], self.N_tot, 3)) for i in range(self.size)]  # create empty dumy on all cores
            self.force_split = [np.empty((self.chunks[i], self.N_tot, 3)) for i in range(self.size)]  # create empty dumy on all cores
            self.steps_split = [np.empty(self.chunks[i]) for i in range(self.size)]
            self.t_split = [np.empty(self.chunks[i]) for i in range(self.size)]

        # Import the correct split position arrays
        self.pos = self.comm.scatter(self.pos_all_split, root=0)
        self.force = self.comm.scatter(self.force_split, root=0)
        self.steps = self.comm.scatter(self.steps_split, root=0)
        self.t = self.comm.scatter(self.t_split, root=0)
        
        self.n_max = len(self.pos[:, 0, 0])  # number of timestep on core
        self.n_max_all = self.comm.allreduce(self.n_max, op=MPI.SUM)
        
        # communicate if cheapened calculation (less interaction types included)
        if self.rank == 0:
            self.cheap = self.cheap
        else:
            self.cheap = False
        self.cheap = self.comm.bcast(self.cheap, root=0)
                    
        if self.rank == 0 and self.verbose is True:
            print('Time after communication', time.time() - self.tstart, flush=True)
             
    def recognize_molecules_all(self, n):
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
            if self.N_OH != self.n_OH[n] or self.n_H3O[n] > 0:  # Exemption for H3O+ cases
                print(f"Strange behaviour found, H3O+ created. Rank", self.rank, 'timestep ', self.t[n])
                print(f"N_H3O+", self.n_H3O[n], flush=True)
                
                # issues = np.setdiff1d(OH_i, self.OH_i[n])
                # for i in issues:
                self.OH_i[n, :] = self.OH_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                
                self.H2O_i[n, :] = self.H2O_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :] + self.L*self.H2O_shift

            elif (OH_i == self.OH_i_s).all():  # No reaction occured only check PBC
                self.OH_i[n, :] = self.OH_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                
                self.H2O_i[n, :] = self.H2O_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :] + self.L*self.H2O_shift

            else:  # Reaction cases
                # find which OH- belongs to which OH-. This is difficult because of sorting differences.
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
                              "CPU rank=", self.rank, 'timestep', self.steps[n], flush=True)

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
                    self.H2O_shift[idx_H2O_i, :] += np.round((real_loc - self.pos_O[n, self.H2O_i[n, idx_H2O_i], :])/self.L).astype(int)  # Correct shifting update for index
                    
                
                # Update all the positions
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :]+ self.L*self.H2O_shift
                self.OH_i_s = OH_i  # always sort after reaction or initiation to have a cheap check lateron.

    def loop_timesteps_all(self, n_samples=10): 
        """This function loops over all timesteps and tracks all over time properties
        
        The function tracks calls the molecule recognition function and the rdf functions when needed.

        Args:
            n_samples (int, optional): time between sampling rdfs. Defaults to 10.
        """
        # split the arrays up to per species description
        self.pos_H = self.pos[:, self.H_i, :]
        self.pos_O = self.pos[:, self.O_i, :]
        self.pos_K = self.pos[:, self.K_i, :]

        # Create per species-species interactions indexing arrays
        self.idx_KK = np.triu_indices(self.N_K, k=1)
        self.idx_KO = np.mgrid[0:self.N_K, 0:self.N_O].reshape(2, self.N_K*self.N_O)
        self.idx_HO = np.mgrid[0:self.N_H, 0:self.N_O].reshape(2, self.N_H*self.N_O)
        self.idx_OO = np.triu_indices(self.N_O, k=1)
        if self.cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
            self.idx_HH = np.triu_indices(self.N_H, k=1)
            self.idx_HK = np.mgrid[0:self.N_H, 0:self.N_K].reshape(2, self.N_H*self.N_K)

        for n in range(self.n_max):  # Loop over all timesteps
            if (n % 10000 == 0) and (self.verbose is True) and (n > 0):
                print("time is:", self.t[n], "rank is:", self.rank, flush=True)
            # Calculate only OH distances for OH- recognition
            r_HO = (self.pos_O[n, self.idx_HO[1], :] - self.pos_H[n, self.idx_HO[0], :] + self.L/2) % self.L - self.L/2
            self.d_HO = np.sqrt(np.sum(r_HO**2, axis=1))
            
            self.recognize_molecules_all(n)
            if n % n_samples == 0:
                # Calculate all other distances for RDF's and such when needed
                self.r_OO = (self.pos_O[n, self.idx_OO[1], :] - self.pos_O[n, self.idx_OO[0], :] + self.L/2) % self.L - self.L/2
                d_OO = np.sqrt(np.sum(self.r_OO**2, axis=1))
                self.d_H2OH2O = d_OO[((np.isin(self.idx_OO[0], self.H2O_i[n])) & (np.isin(self.idx_OO[1], self.H2O_i[n])))]
                self.d_OHH2O = d_OO[(((np.isin(self.idx_OO[0], self.H2O_i[n])) & (np.isin(self.idx_OO[1], self.OH_i[n])))) |
                                    (((np.isin(self.idx_OO[0], self.OH_i[n])) & (np.isin(self.idx_OO[1], self.H2O_i[n]))))]  # ((a and b) or (b and a)) conditional
                
                r_KO = (self.pos_O[n, self.idx_KO[1], :] - self.pos_K[n, self.idx_KO[0], :] + self.L/2) % self.L - self.L/2
                d_KO = np.sqrt(np.sum(r_KO**2, axis=1))
                self.d_KOH = d_KO[np.isin(self.idx_KO[1], self.OH_i[n])]  # selecting only OH from array
                self.d_KH2O = d_KO[np.isin(self.idx_KO[1], self.H2O_i[n])]  # selecting only OH from array

                if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
                    self.d_OHOH = d_OO[((np.isin(self.idx_OO[0], self.OH_i[n])) & (np.isin(self.idx_OO[1], self.OH_i[n])))]
                    self.r_KK = (self.pos_K[n, self.idx_KK[1], :] - self.pos_K[n, self.idx_KK[0], :] + self.L/2) % self.L - self.L/2
                    self.d_KK = np.sqrt(np.sum(self.r_KK**2, axis=1))

                if self.cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
                    self.d_HOH = self.d_HO[np.isin(self.idx_HO[1], self.OH_i[n])]
                    self.d_HH2O = self.d_HO[np.isin(self.idx_HO[1], self.H2O_i[n])]
                    
                    r_HH = (self.pos_H[n, self.idx_HH[1], :] - self.pos_H[n, self.idx_HH[0], :] + self.L/2) % self.L - self.L/2
                    self.d_HH = np.sqrt(np.sum(r_HH**2, axis=1))
                    
                    r_HK = (self.pos_K[n, self.idx_HK[1], :] - self.pos_H[n, self.idx_HK[0], :] + self.L/2) % self.L - self.L/2
                    self.d_HK = np.sqrt(np.sum(r_HK**2, axis=1))
                    
                    self.d_KO_all = d_KO
                    self.d_OO_all = d_OO

                # Now compute RDF results
                self.rdf_compute_all(n)
                # self.rdf_force_compute_all(n)  # enter the force rdf
            

        if self.rank == 0 and self.verbose is True:
            print('Time calculating distances', time.time() - self.tstart)

    def rdf_compute_all(self, n, nb=128, r_max=None, force_RDF=False):
        # RDF startup scheme
        if n == 0:
            # set standard maximum rdf value
            r_min = 1
            if r_max == None:
                r_max = self.L/2  # np.sqrt(3*self.L**2/4)  # set to default half box diagonal distance

            self.rdf_sample_counter = 0
            # set basic properties
            self.r = np.histogram(self.d_H2OH2O, bins=nb + 1, range=(r_min, r_max))[1] # array with outer edges
            
            # Standard rdf pairs
            self.rdf_H2OH2O = np.zeros(self.r.size - 1, dtype=float)
            self.rdf_OHH2O = np.zeros_like(self.rdf_H2OH2O)
            self.rdf_KOH = np.zeros_like(self.rdf_H2OH2O)
            self.rdf_KH2O = np.zeros_like(self.rdf_H2OH2O)
            if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
                self.rdf_OHOH = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_KK = np.zeros_like(self.rdf_H2OH2O)
            if self.cheap is False: # Also execute Hydrogen interaction distances (long lists)
                self.rdf_HOH = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_HH2O = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_HK = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_HH = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_KO_all = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_OO_all = np.zeros_like(self.rdf_H2OH2O)

        # Now calculate all rdf's (without rescaling them, will be done later)
        self.rdf_H2OH2O += np.histogram(self.d_H2OH2O, bins=self.r)[0]
        self.rdf_OHH2O += np.histogram(self.d_OHH2O, bins=self.r)[0]
        self.rdf_KOH += np.histogram(self.d_KOH, bins=self.r)[0]
        self.rdf_KH2O += np.histogram(self.d_KH2O, bins=self.r)[0]

        if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
            self.rdf_OHOH += np.histogram(self.d_OHOH, bins=self.r)[0]
            self.rdf_KK += np.histogram(self.d_KK, bins=self.r)[0]
    
        if self.cheap is False: # Also execute Hydrogen interaction distances (long lists)
                self.rdf_HOH += np.histogram(self.d_HOH, bins=self.r)[0]
                self.rdf_HH2O += np.histogram(self.d_HH2O, bins=self.r)[0]
                self.rdf_HK += np.histogram(self.d_HK, bins=self.r)[0]
                self.rdf_HH += np.histogram(self.d_HK, bins=self.r)[0]
                self.rdf_KO_all += np.histogram(self.d_KO_all, bins=self.r)[0]
                self.rdf_OO_all += np.histogram(self.d_OO_all, bins=self.r)[0]
        
        self.rdf_sample_counter += 1

    def rdf_force_compute_all(self, n, nb=256, r_max=None, force_RDF=False):
        # RDF startup scheme
        if n == 0:
            # Split up the forces when needed
            self.F_H = self.force[:, self.H_i, :]
            self.F_O = self.force[:, self.O_i, :]
            self.F_K = self.force[:, self.K_i, :]
            
            # set standard maximum rdf value
            r_min = 1
            if r_max == None:
                r_max = self.L/2  # np.sqrt(3*self.L**2/4)  # set to default half box diagonal distance

            self.f_rdf_sample_counter = 0
            # set basic properties
            self.f_r = np.histogram(self.d_H2OH2O, bins=nb, range=(r_min, r_max))[1] # array with outer edges
            
            # Standard rdf pairs
            self.f_rdf_H2OH2O = np.zeros(self.f_r.size, dtype=float)
            self.f_rdf_OHH2O = np.zeros_like(self.f_rdf_H2OH2O)
            self.f_rdf_KOH = np.zeros_like(self.f_rdf_H2OH2O)
            self.f_rdf_KH2O = np.zeros_like(self.f_rdf_H2OH2O)
            if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
                self.f_rdf_OHOH = np.zeros_like(self.f_rdf_H2OH2O)
                self.f_rdf_KK = np.zeros_like(self.f_rdf_H2OH2O)
            if self.cheap is False: # Also execute Hydrogen interaction distances (long lists)
                self.f_rdf_HOH = np.zeros_like(self.f_rdf_H2OH2O)
                self.f_rdf_HH2O = np.zeros_like(self.f_rdf_H2OH2O)
                self.f_rdf_HK = np.zeros_like(self.f_rdf_H2OH2O)
                self.f_rdf_HH = np.zeros_like(self.f_rdf_H2OH2O)
                self.f_rdf_KO_all = np.zeros_like(self.f_rdf_H2OH2O)
                self.f_rdf_OO_all = np.zeros_like(self.f_rdf_H2OH2O)

        # When i and j are of the same species with N particles
        # every iteration F_rdf += sum_j=0->N[ sum_i=j+1->N  ( vec_F_j dot vec_r_ij * (1/r_ij^3) * Heaviside(r - r_ij) ) ]
        # Need to calculate after all iterations are done and stiching together
        # F_rdf *= L^3/(2*pi*kB*T*N^2)
    
        # When i and j are of different species N_1 and N_2
        # every iteration F_rdf+= sum_j=0->N_1 [ sum_j=0->N_2 ( vec_F_j dot vec_r_ij * (1/r_ij^3) * Heaviside(r - r_ij) )  ]
        # Need to calculate after all iterations are done and stiching together
        # F_rdf *= L^3/(2*pi*kB*T*N_1 * N_2)
        
        # Test with f_rdf_KK as test array
        F_KK = self.F_K[n, self.idx_KK[0], :]  # make array same size as d_KK by repeating using idx_KK array 
        self.r_rdf_KK += np.einsum('ij,ij->i', self.F_K[n, self.idx_KK[0], :], self.r_KK)/(np.power(self.d_KK, 3))*np.where(r < self.d_KK, 1, 0)  # computes vec_F_j dot vec_r_ij * 1/r_ij^3
        # Ideas for later
        r_H2OH2O = self.r_OO[((np.isin(self.idx_OO[0], self.H2O_i[n])) & (np.isin(self.idx_OO[1], self.H2O_i[n]))), :]
        F_H2O = self.force[n, ]
        
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
        
        # prepair gethering on all cores 1) All H3O+ stuf   f
        self.n_H3O = self.comm.gather(self.n_H3O, root=0)
        
        self.K = self.comm.gather(self.pos_K, root=0)
        self.H = self.comm.gather(self.pos_H, root=0)

        # gather the time arrays as well
        self.t = self.comm.gather(self.t, root=0)
        
        # RDF's
        # First rescale all RDFs accordingly also prepaire for averaging using mpi.sum
        self.r_cent = (self.r[:-1] + self.r[1:])/2  # central point of rdf bins
        rdf_sample_counter_all = self.comm.allreduce(self.rdf_sample_counter, op=MPI.SUM)
        rescale_geometry = (4*np.pi*self.r_cent**2)*(self.r[1] - self.r[0])  # 4*pi*r^2*dr
        rescale_counters = (self.L**3)/(rdf_sample_counter_all)
        rescale = rescale_counters/rescale_geometry
        
        self.rdf_H2OH2O *= rescale/(self.N_H2O*(self.N_H2O - 1)*0.5)  # rdf*L_box^3/(n_sample*n_interactions*geometry_rescale/n_cores)
        self.rdf_OHH2O *= rescale/(self.N_OH*self.N_H2O)
        self.rdf_KOH *= rescale/(self.N_OH*self.N_K)
        self.rdf_KH2O *= rescale/(self.N_H2O*self.N_K)
        if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
            self.rdf_OHOH *= rescale/(self.N_OH*(self.N_OH - 1)*0.5)  
            self.rdf_KK *= rescale/(self.N_K*(self.N_K - 1)*0.5)
        if self.cheap is False:
            self.rdf_HOH *= rescale/(self.N_H*self.N_OH)
            self.rdf_HH2O *= rescale/(self.N_H*self.N_H2O)
            self.rdf_HK *= rescale/(self.N_H*self.N_K)
            self.rdf_HH *= rescale/(self.N_H*(self.N_H - 1)*0.5)
            self.rdf_KO_all *= rescale/(self.N_K*self.N_O)
            self.rdf_OO_all *= rescale/(self.N_O*(self.N_O - 1)*0.5)
        
        # Then communicate these to main core.
        self.rdf_H2OH2O = self.comm.reduce(self.rdf_H2OH2O, op=MPI.SUM)
        self.rdf_OHH2O = self.comm.reduce(self.rdf_OHH2O, op=MPI.SUM, root=0)       
        self.rdf_KOH = self.comm.reduce(self.rdf_KOH, op=MPI.SUM, root=0)
        self.rdf_KH2O = self.comm.reduce(self.rdf_KH2O, op=MPI.SUM, root=0)

        if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
            self.rdf_OHOH = self.comm.reduce(self.rdf_OHOH, op=MPI.SUM, root=0)
            self.rdf_KK = self.comm.reduce(self.rdf_KK, op=MPI.SUM, root=0)
        
        if self.cheap is False:
            self.rdf_HOH = self.comm.reduce(self.rdf_HOH, op=MPI.SUM, root=0)
            self.rdf_HH2O = self.comm.reduce(self.rdf_HH2O, op=MPI.SUM, root=0)
            self.rdf_HK = self.comm.reduce(self.rdf_HK, op=MPI.SUM, root=0)
            self.rdf_HH = self.comm.reduce(self.rdf_HH, op=MPI.SUM, root=0)
            self.rdf_KO_all = self.comm.reduce(self.rdf_KO_all, op=MPI.SUM, root=0)
            self.rdf_OO_all = self.comm.reduce(self.rdf_OO_all, op=MPI.SUM, root=0)

        # Stich together correctly on main cores
        if self.rank == 0:
            self.stitching_together_main()
      
    def stitching_together_main(self):
        # NOW ADD 1 reaction recognition and 2 reordering
        # for every end of 1 section check with start next one
        for n in range(self.size-1):
            # OH-
            OH_not_found = np.empty(0)
            mismatch_indices = np.where(self.OH_i[n][-1, :] != self.OH_i[n+1][0, :])[0]
            # Swap columns in C_modified based on the mismatch
            for index in mismatch_indices:
                # Find the index in B corresponding to the value in A
                b_index = np.where(self.OH_i[n+1][0, :] == self.OH_i[n][-1, index])[0]
                if b_index.size == 0:
                    OH_not_found = np.append(OH_not_found, self.OH_i[n][-1, index])
                    if self.verbose is True:
                        print('CHECK: OH part Reaction occured on stitching back together', flush=True)

                else:
                    self.OH_i[n+1][:, [index, b_index[0]]] = self.OH_i[n+1][:, [b_index[0], index]]
                    self.OH[n+1][:, [index, b_index[0]], :] = self.OH[n+1][:, [b_index[0], index], :]
                    self.OH_shift[n+1][[index, b_index[0]], :] = self.OH_shift[n+1][[b_index[0], index], :]
            
            # H2O
            H2O_not_found = np.empty(0)
            mismatch_indices = np.where(self.H2O_i[n][-1, :] != self.H2O_i[n+1][0, :])[0]
            # Swap columns in C_modified based on the mismatch
            for index in mismatch_indices:
                # Find the index in B corresponding to the value in A
                b_index = np.where(self.H2O_i[n+1][0, :] == self.H2O_i[n][-1, index])[0]
                if b_index.size == 0:
                    H2O_not_found = np.append(H2O_not_found, self.H2O_i[n][-1, index])
                    if self.verbose is True:
                        print('CHECK: H2O part Reaction occured on stitching back together', flush=True)
                else:
                    self.H2O_i[n+1][:, [index, b_index[0]]] = self.H2O_i[n+1][:, [b_index[0], index]]
                    self.H2O[n+1][:, [index, b_index[0]], :] = self.H2O[n+1][:, [b_index[0], index], :]
                    self.H2O_shift[n+1][[index, b_index[0]], :] = self.H2O_shift[n+1][[b_index[0], index], :]
            
            #### these two for loops only activate if there are reactions at stichting location
            idx_H2O = np.zeros_like(H2O_not_found, dtype=int)
            for i, H2O_i in enumerate(H2O_not_found):
                idx_H2O[i] = np.where(self.H2O_i[n][-1, :] == H2O_i)[0][0]

            for OH_old in OH_not_found:
                # OH part
                idx_OH_old = np.where(self.OH_i[n][-1, :] == OH_old)[0][0]

                # find closest H2O for reaction in the index_H2O lists
                r_OO = (self.OH[n][-1, idx_OH_old] - self.H2O[n][-1, idx_H2O] + self.L/2) % self.L - self.L/2
                d2 = np.sum(r_OO**2, axis=1)
                i = np.argmin(d2) # index in the H2O not found array
                if d2[i] > 9:  # excape when jump too large
                        print("Strange behaviour found, reaction jump too far, d = ", np.sqrt(d2[i]), 'Angstrom',
                                        "during stitching back together", flush=True)
                H2O_old = H2O_not_found[i]
                idx_H2O_old = idx_H2O[i]
                
                OH_new = H2O_old
                H2O_new = OH_old
                
                idx_OH_new = np.where(self.OH_i[n+1][0, :] == OH_new)[0][0]
                idx_H2O_new = np.where(self.H2O_i[n+1][0, :] == H2O_new)[0][0]
                # check if index of OH_new is already the right one
                
                if self.verbose is True:
                    print('matching up OH', OH_old, OH_new, idx_OH_old, idx_OH_new)
                    print('matching up H2O', H2O_old, H2O_new, idx_H2O_old, idx_H2O_new)
                if idx_OH_new != idx_OH_old: ## if not, swap around
                    self.OH_i[n+1][:, [idx_OH_old, idx_OH_new]] = self.OH_i[n+1][:, [idx_OH_new, idx_OH_old]]
                    self.OH[n+1][:, [idx_OH_old, idx_OH_new], :] = self.OH[n+1][:, [idx_OH_new, idx_OH_old], :]
                    self.OH_shift[n+1][[idx_OH_old, idx_OH_new], :] = self.OH_shift[n+1][[idx_OH_new, idx_OH_old], :]
                
                if idx_H2O_new != idx_H2O_old: ## if not, swap around
                    self.H2O_i[n+1][:, [idx_H2O_new, idx_H2O_new]] = self.H2O_i[n+1][:, [idx_H2O_new, idx_H2O_old]]
                    self.H2O[n+1][:, [idx_H2O_old, idx_H2O_new], :] = self.H2O[n+1][:, [idx_H2O_new, idx_H2O_old], :]
                    self.H2O_shift[n+1][[idx_H2O_old, idx_H2O_new], :] = self.H2O_shift[n+1][[idx_H2O_new, idx_H2O_old], :]
                # so now we now for sure that idx_H2O_old=idx_H2O_new and index_OH_old=index_OH_new
                # and that H2O_old=OH_new and that OH_old=H2O_new
                
                # Only check for pbc during a reaction for the OH
                dis = (self.OH[n+1][0, idx_OH_old, :] - self.OH[n][-1, idx_OH_old, :] + self.L/2) % self.L - self.L/2
                real_loc = self.OH[n][-1, idx_OH_old, :] + dis
                self.OH_shift[n][idx_OH_old, :] = np.round((real_loc - self.OH[n+1][0, idx_OH_old, :])/self.L).astype(int)
                
                # Only check for pbc during a reaction for the H2O
                dis = (self.H2O[n+1][0, idx_H2O_old, :] - self.H2O[n][-1, idx_H2O_old, :] + self.L/2) % self.L - self.L/2
                real_loc = self.H2O[n][-1, idx_H2O_old, :] + dis
                self.H2O_shift[n][idx_H2O_old, :] = np.round((real_loc - self.H2O[n+1][0, idx_H2O_old, :])/self.L).astype(int)
            #### Till here

            # Now adjust for passing PBC in the shift collumns. OH-
            self.OH[n+1][:, :, :] += self.L*self.OH_shift[n][:, :]
            self.OH_shift[n+1][:, :] += self.OH_shift[n][:, :]
            
            # Now adjust for passing PBC in the shift collumns. H2O
            self.H2O[n+1][:, :, :] += self.L*self.H2O_shift[n][:, :]
            self.H2O_shift[n+1][:, :] += self.H2O_shift[n][:, :]
            
        # Combining adjusted arrays back to the right shape
        # OH-
        self.OH = np.concatenate(self.OH, axis=0)
        self.OH_i = np.concatenate(self.OH_i, axis=0)
        self.n_OH = np.concatenate(self.n_OH, axis=0)
        
        # H2O
        self.H2O = np.concatenate(self.H2O, axis=0)
        self.H2O_i = np.concatenate(self.H2O_i, axis=0)
        self.n_H2O = np.concatenate(self.n_H2O, axis=0)
        
        # H3O+
        self.n_H3O = np.concatenate(self.n_H3O, axis=0)
        
        # K+
        self.K = np.concatenate(self.K, axis=0)
        
        # H
        self.H = np.concatenate(self.H,  axis=0)

        # RDF functionality has no need to do anything
        
        # Getting a time array
        self.t = np.concatenate(self.t, axis=0)

    def compute_MSD(self):
        # prepaire windowed MSD calculation mode with freud
        msd = freud.msd.MSD(mode='window')
        
        self.msd_OH = msd.compute(self.OH).msd
        self.msd_H2O = msd.compute(self.H2O).msd
        self.msd_K = msd.compute(self.K).msd

    def save_results_all(self):
        # separate single core or multi core folders
        if self.size == 1:
            path = self.folder + r"/single_core/"
        else:
            path = self.folder
        self.save_numpy_files_main(path)
        
        # create the output.h5 file always on main core
        if self.rank == 0:
            self.create_dataframe_main(path)
        
        # save position output if needed as .xyz (4 seperate files) on separate cores
        # communicate arrays
        if self.xyz_out is True:
            if self.rank == 0:
                shape = self.pos_all.shape
                self.pos_pro = np.concatenate((self.OH, self.H2O, self.K, self.H), axis=1)
            else:
                shape = None
            
            shape = self.comm.bcast(shape, root=0)
            
            if self.rank !=0:
                self.pos_all = np.empty(shape)
                self.pos_pro = np.empty(shape)
            
            self.pos_pro = self.comm.bcast(self.pos_pro, root=0)
            self.pos_all = self.comm.bcast(self.pos_all, root=0)  
            for i in range(4):
                if (i+1) % self.size == self.rank:
                    self.write_to_xyz_all(i)

    def create_dataframe_main(self, path):
        # create large dataframe with output
        df = h5py.File(path + '/output.h5', "w")
        
        # rdfs
        df.create_dataset("rdf/r", data=self.r_cent)
        df.create_dataset("rdf/g_H2OH2O(r)", data=self.rdf_H2OH2O)
        df.create_dataset("rdf/g_OHH2O(r)", data=self.rdf_OHH2O)
        df.create_dataset("rdf/g_KH2O(r)", data=self.rdf_KH2O)
        
        if self.N_K > 1:
            df.create_dataset("rdf/g_OHOH(r)", data=self.rdf_OHOH)
            df.create_dataset("rdf/g_KK(r)", data=self.rdf_KK)

        if self.cheap is False:
            df.create_dataset("rdf/g_HOH(r)", data=self.rdf_HOH)
            df.create_dataset("rdf/g_HK(r)", data=self.rdf_HK)
            df.create_dataset("rdf/g_HH(r)", data=self.rdf_HH)
            df.create_dataset("rdf/g_KO(r)", data=self.rdf_KO_all)
            df.create_dataset("rdf/g_OO(r)", data=self.rdf_OO_all)
        
        # msds
        df.create_dataset("msd/OH", data=self.msd_OH)
        df.create_dataset("msd/H2O", data=self.msd_H2O)
        df.create_dataset("msd/K", data=self.msd_K)
        
        # properties over time
        df.create_dataset("transient/time", data=self.t)
        df.create_dataset("transient/index_OH", data=self.OH_i)
        df.create_dataset("transient/index_H2O", data=self.H2O_i)
        df.create_dataset("transient/index_K", data=self.K_i)
        df.create_dataset("transient/pos_OH", data=self.OH)
        df.create_dataset("transient/pos_H2O", data=self.H2O)
        df.create_dataset("transient/pos_K", data=self.K)
        df.create_dataset("transient/stresses", data=self.stress)
        df.create_dataset("transient/energies", data=self.energy)
        df.close()

    def write_to_xyz_all(self, type):
        # assesing tasks correctly
        if type == 0:
            ## unwrapped unprocessed postitions
            types = ['H']*self.N_H + ['O']*self.N_O + ['K']*self.N_K
            pos = self.pos_all
            name = '/traj_unprocessed_unwrapped.xyz'
        if type == 1:
            ## wrapped processed postitions
            types = ['H']*self.N_H + ['O']*self.N_O + ['K']*self.N_K
            pos = self.pos_all % self.L
            name = '/traj_unprocessed_wrapped.xyz'
        if type == 2:
            ## unwrapped unprocessed postitions
            types = ['F']*self.N_OH + ['O']*self.N_H2O + ['K']*self.N_K + ['H']*self.N_H
            pos = self.pos_pro
            name = '/traj_processed_unwrapped.xyz'
        if type == 3:
            ## wrapped processed postitions
            types = ['F']*self.N_OH + ['O']*self.N_H2O + ['K']*self.N_K + ['H']*self.N_H
            pos = self.pos_pro % self.L
            name = '/traj_processed_wrapped.xyz'

        if self.verbose is True:
            print(f'prepare for writing {name} started on rank: {self.rank}')
        
        configs = [None]*pos.shape[0]
        for i in range(self.pos_all.shape[0]):
            configs[i] = ase.Atoms(types, positions=pos[i, :, :], cell=[self.L, self.L, self.L], pbc=True)
        ase.io.write(os.path.normpath(self.folder+name), configs, format='xyz', parallel=False)

        if self.verbose is True:
            print(f'writing {self.folder+name} completed on rank: {self.rank}')

    def save_numpy_files_main(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            # print("os.mkdir threw error, but continued with except:", error)
            error = 1
        if self.cheap is False:
            np.savez(path + r"/output.npz",
                    OH_i=self.OH_i, OH=self.OH, H2O_i=self.H2O_i, H2O=self.H2O,  # tracking OH-
                    r_rdf=self.r_cent, rdf_H2OH2O=self.rdf_H2OH2O, rdf_OHH2O=self.rdf_OHH2O, rdf_KH2O=self.rdf_KH2O,
                    rdf_HOH=self.rdf_HOH, rdf_HK=self.rdf_HK, rdf_HH=self.rdf_HH, rdf_KO_all=self.rdf_KO_all, rdf_OO_all=self.rdf_OO_all)  # sensing the rdf
        else:
            np.savez(path + r"/output.npz",
                    OH_i=self.OH_i, OH=self.OH, H2O_i=self.H2O_i, H2O=self.H2O,  # tracking OH-
                    r_rdf=self.r_cent, rdf_H2OH2O=self.rdf_H2OH2O, rdf_OHH2O=self.rdf_OHH2O, rdf_KH2O=self.rdf_KH2O)  # sensing the rdf
            
    def test_combining_main(self):
        if self.size == 1:                
            path = self.folder + r"/single_core"
            try:
                os.mkdir(path)
            except OSError as error:
                error = 1
            
        else:
            path = self.folder + r"/multi_core"
            try:
                os.mkdir(path)
            except OSError as error:
                error = 1
            try:
                loaded = np.load(self.folder + r"/single_core/output.npz")
                # test outputs of code
                if np.allclose(self.OH_i, loaded['OH_i'], rtol=1e-05, atol=1e-08, equal_nan=False) == False:
                    print("Indexing OH between multicore and single core arrays differs more than acceptable")
                if np.allclose(self.OH, loaded['OH'], rtol=1e-05, atol=1e-08, equal_nan=False) == False:
                    print("Positions OH between multicore and single core arrays differs more than acceptable")
                if np.allclose(self.H2O_i, loaded['H2O_i'], rtol=1e-05, atol=1e-08, equal_nan=False) == False:
                    print("Indexing H2O between multicore and single core arrays differs more than acceptable")
                if np.allclose(self.H2O, loaded['H2O'], rtol=1e-05, atol=1e-08, equal_nan=False) == False:
                    print("Positions H2O between multicore and single core arrays differs more than acceptable")
                if np.allclose(self.rdf_H2OH2O, loaded['rdf_H2OH2O'], rtol=1e-05, atol=1e-08, equal_nan=False) == False:
                    print("RDF H2O-H2O between multicore and single core arrays differs more than acceptable")
                    print("maximum difference =", np.max(self.rdf_H2OH2O - loaded['rdf_H2OH2O']), loaded['r_rdf'][np.argmax(self.rdf_H2OH2O - loaded['rdf_H2OH2O'])])
                    plt.plot(self.r_cent, np.abs(self.rdf_H2OH2O - loaded['rdf_H2OH2O']), label='water - water')
                if np.allclose(self.rdf_OHH2O, loaded['rdf_OHH2O'], rtol=1e-05, atol=1e-08, equal_nan=False) == False:
                    print("RDF OH-H2O between multicore and single core arrays differs more than acceptable")
                    print("maximum difference =", np.max(self.rdf_OHH2O - loaded['rdf_OHH2O']), loaded['r_rdf'][np.argmax(self.rdf_OHH2O - loaded['rdf_OHH2O'])])
                    plt.plot(self.r_cent, np.abs(self.rdf_OHH2O - loaded['rdf_OHH2O']), label='hydroxide - water')
                if np.allclose(self.rdf_KH2O, loaded['rdf_KH2O'], rtol=1e-05, atol=1e-08, equal_nan=False) == False:
                    print("RDF K-H2O between multicore and single core arrays differs more than acceptable")
                    print("maximum difference =", np.max(self.rdf_KH2O - loaded['rdf_KH2O']), loaded['r_rdf'][np.argmax(self.rdf_KH2O - loaded['rdf_KH2O'])])
                    plt.plot(self.r_cent, np.abs(self.rdf_KH2O - loaded['rdf_KH2O']), label='potassium - water')
                    plt.savefig(path + r'/rdf_diff.png')
                    plt.close()
            except:
                print('No single core checking file availlable, checking is useless', flush=True)
            
            # plt.figure()
            # # plt.plot(self.OH[:, 0, :], label=['x', 'y', 'z'])
            # plt.plot(self.t, self.OH_i)
            # for i in range(1, self.size):
            #     plt.axvline(x=self.t[-1]*i/self.size, color = 'c')
            # plt.xlabel('time/[fs]')      
            # plt.ylabel('OH- atom index')          
            # plt.savefig(os.path.normpath(path + r'/index_OH.png'))
            # plt.close()
            
            # plt.figure()
            # plt.plot(self.t, self.n_OH)
            # for i in range(1, self.size):
            #     plt.axvline(x=self.t[-1]*i/self.size, color = 'c')
            # plt.xlabel('time/[fs]')      
            # plt.ylabel('number of OH-')
            # plt.savefig(os.path.normpath(path + r'/n_OH.png'))
            # plt.close()
            
            # disp = np.sqrt(np.sum((self.OH[1:, :, :]- self.OH[:-1, :, :])**2, axis=2))
            # plt.figure()
            # plt.plot(self.t[:-1], disp)
            # # for i in range(1, self.size):
            # #     plt.axvline(x=self.t[-1]*i/self.size, color = 'c')
            # plt.xlabel('time/[fs]')      
            # plt.ylabel('OH displacement between timesteps/[Angstrom]')
            # plt.savefig(os.path.normpath(path + r'/dis_OH.png'))
            # plt.close()
            
            # disp = np.sqrt(np.sum((self.H2O[1:, :, :]- self.H2O[:-1, :, :])**2, axis=2))
            # plt.figure()
            # plt.plot(self.t[:-1], disp)
            # # for i in range(1, self.size):
            # #     plt.axvline(x=self.t[-1]*i/self.size, color = 'c')
            # plt.xlabel('time/[fs]')      
            # plt.ylabel('H2O displacement between timesteps/[Angstrom]')
            # plt.savefig(os.path.normpath(path + r'/dis_H2O.png'))
            # plt.close()
            
            # plt.figure()
            # plt.plot(self.r_cent, self.rdf_H2OH2O, label='Water-Water')
            # plt.plot(self.r_cent, self.rdf_KH2O, label='Potassium-Water')
            # plt.plot(self.r_cent, self.rdf_OHH2O, label='Hydroxide-Water')
            # plt.legend()
            # plt.xlabel('radius in A')
            # plt.ylabel('g(r)')
            # plt.savefig(os.path.normpath(path + r'/rdf_H2OH2O'))
            # plt.close()


### TEST LOCATIONS ###
# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/")
# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1")
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1")
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/")
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/", verbose=True)
# Traj1 = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/combined_simulation/", cheap=True, xyz_out=True, verbose=True)
# Traj2 = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/longest_up_till_now/", cheap=True, xyz_out=True, verbose=True)
Traj3 = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/1ns/", cheap=True, xyz_out=True, verbose=True)

# Traj = Prot_Hop(r"./", cheap=True, xyz_out=True, verbose=True)
