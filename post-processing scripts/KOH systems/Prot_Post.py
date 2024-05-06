import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

class Prot_Post:
    def __init__(self, folder):
        self.folder = folder
        self.load_properties()

    def load_properties(self):
        input = h5py.File(os.path.normpath(self.folder+'output.h5'), 'r')

        # Retrieving the msd properties
        self.msd_OH = input["msd/OH"][()]
        self.msd_H2O = input["msd/H2O"][()]
        self.msd_K = input["msd/K"][()]
        
        # retrieving the transient properties
        self.t = input["transient/time"][()]
        self.stress = input["transient/stresses"][()]
        self.energy = input["transient/energies"][()]
        
        self.index_OH = input["transient/index_OH"][()]
        self.index_K = input["transient/index_K"][()]
        self.index_H2O = input["transient/index_H2O"][()]
        
        self.OH = input["transient/pos_OH"][()]
        self.K = input["transient/pos_K"][()]
        self.H2O = input["transient/pos_H2O"][()]
        
        self.N_OH = self.OH.shape[1]
        self.N_K = self.K.shape[1]
        self.N_H2O = self.H2O.shape[1]
        self.N_m = np.array([self.N_OH, self.N_K, self.N_H2O], dtype=int)  # count of molecule types OH- K+ H2O
        self.N_m_n = np.array(['OH', 'K', 'H2O'], dtype=str)
        self.N = np.array([self.N_OH + 2*self.N_H2O, self.N_OH + self.N_H2O, self.N_K], dtype=int)  # count of atom types, H, O, K
        self.N_n = np.array(['H', 'K', 'K'], dtype=str)
        
         # load rdfs
        self.rdf_r = input["rdf/r"][()]
        self.rdf_H2OH2O = input["rdf/g_H2OH2O(r)"][()]
        self.rdf_OHH2O = input["rdf/g_OHH2O(r)"][()]
        self.rdf_KH2O = input["rdf/g_KH2O(r)"][()]
        
        if self. N_K > 1:
            self.rdf_OHOH = input["rdf/g_OHOH(r)"][()]        
            self.rdf_KK = input["rdf/g_KK(r)"][()]
    
        try:
            self.rdf_HOH = input["rdf/g_HOH(r)"][()]
            self.cheap = False
        except:
            self.cheap = True
        
        if self.cheap is False:
            self.rdf_HK = input["rdf/g_HK(r)"][()]
            self.rdf_HH = input["rdf/g_HH(r)"][()]
            self.rdf_KO_all = input["rdf/g_KO(r)"][()]
            self.rdf_OO_all = input["rdf/g_OO(r)"][()]
        input.close()
    
    def diffusion(self, specie, t_start=1000, steps=2000, m=False, plotting=False):
        # Settings for the margins and fit method.
        margin = 0.005  # cut away range at left and right side
        Minc = 125  # minimum number of points included in the fit
        Mmax = 250  # maximum number of points included in the fit
        er_max = 0.1  # maximum allowed error
        
        try:
            MSD_in = getattr(self, f"msd_{specie}")
        except:
            ValueError("Input variable indicates non existing specie")
        
        n = np.logspace(np.log10(t_start), np.log10(self.t[-1]), steps, dtype=int)
        n = n[:np.where(n > self.t.shape)[0][0]]  # make very sure to not overflow time array
        
        t = self.t[n]*1e-15
        MSD_in = MSD_in[n]

        t_log = np.log10(t)
        MSD_log_in = np.log10(np.abs(MSD_in))
        ibest = 'failed'
        jbest = 'failed'
        mbest = 0

        for i in range(int(margin*len(t_log)), int((1-margin)*len(t_log))-Minc):
            for j in range(Minc, min(Mmax, int((1-margin)*len(t_log))-Minc-i)):
                if (t[i] != t[i+1]):
                    p, res, aa, aa1, aa3 = np.polyfit(t_log[i:i+j],
                                                    MSD_log_in[i:i+j], 1,
                                                    full=True)
                    mlog = p[0]
                    if (mlog > (1-er_max) and mlog < (1+er_max) and abs(mbest-1) > abs(mlog-1)):
                        mbest = mlog
                        jbest = j
                        ibest = i

        # Make sure to return NaN (not included in np.nanmean() for averaging).
        if ibest == 'failed':
            D = np.nan
            t_fit = t[0]
            fit = MSD_in[0]

        else:
            D, b = np.polyfit(t[ibest:ibest+jbest],
                            MSD_in[ibest:ibest+jbest], 1)

            # Test box size to displacement comparison.
            if np.abs(MSD_in[ibest+jbest]-MSD_in[ibest]) < m**2 and type(m) is not bool:
                print('MSD fit is smaller than simulation box',
                    MSD_in[ibest+jbest]-MSD_in[ibest], 'versus', m**2)

            t_fit = t[ibest:ibest+jbest]
            fit = D*t_fit + b

        if plotting is True:
            plt.figure('Diffusion fitting')
            plt.loglog(t, MSD_in, 'o', label='data')
            plt.loglog(t_fit, fit, '-.', label='fit')
            plt.grid()
            plt.legend()

        fact = (1e-20)/(6)
        return D*fact
        

# folder = r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/1ns/"

# post = Prot_Post(folder)