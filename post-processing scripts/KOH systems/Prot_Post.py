import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from typing import Tuple

class Prot_Post:
    def __init__(self, folder):
        self.folder = os.path.normpath(folder)
        if (os.name == 'posix') and ('WSL_DISTRO_NAME' in os.environ):
            # Convert Windows path to WSL path
            self.folder = os.path.normpath('/mnt/c'+self.folder)
        elif (os.name == 'posix') and ('delftblue' == os.environ['CMD_WLM_CLUSTER_NAME']):
            # Convert to fully relative paths
            self.folder = os.path.join(os.getcwd(), self.folder)
        elif (os.name == 'nt') and ('vlagerweij == os.getlogin()'):
            # use standard windows file path
            self.folder = self.folder
        else:
            print('no automatic filepath conversiona evailable use relative path')
            self.folder = os.getcwd()
        self.load_properties()

    def load_properties(self):
        input = h5py.File(os.path.normpath(self.folder+'/output.h5'), 'r')

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
        self.rdf_KOH = input["rdf/g_KOH(r)"][()]
        
        if self. N_K > 1:
            self.rdf_OHOH = input["rdf/g_OHOH(r)"][()]        
            self.rdf_KK = input["rdf/g_KK(r)"][()]
    
        try:
            self.rdf_HOH = input["rdf/g_HOH(r)"][()]
            self.cheap = False
        except:
            self.cheap = True
        
        if self.cheap is False:
            self.rdf_HH2O = input["rdf/g_HH2O(r)"][()]
            # self.rdf_HK = input["rdf/g_HK(r)"][()]
            self.rdf_HH = input["rdf/g_HH(r)"][()]
            self.rdf_KO_all = input["rdf/g_KO(r)"][()]
            self.rdf_OO_all = input["rdf/g_OO(r)"][()]
        
        # retrieve the force rdfs
        self.rdf_F_r = input["rdf_F/r"][()]
        self.rdf_F_H2OH2O = input["rdf_F/g_H2OH2O(r)"][()]
        self.rdf_F_OHH2O = input["rdf_F/g_OHH2O(r)"][()]
        self.rdf_F_KOH = input["rdf_F/g_KOH(r)"][()]
        self.rdf_F_KH2O = input["rdf_F/g_KH2O(r)"][()]
        if self. N_K > 1:
            self.rdf_F_OHOH = input["rdf_F/g_OHOH(r)"][()]        
            self.rdf_F_KK = input["rdf_F/g_KK(r)"][()]
        if self.cheap is False:
            self.rdf_F_HOH = input["rdf_F/g_HOH(r)"][()]
            self.rdf_F_HH2O = input["rdf_F/g_HH2O(r)"][()]
            self.rdf_F_HH = input["rdf_F/g_HH(r)"][()]
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
        
        # n = np.logspace(np.log10(t_start), np.log10(self.t[-1]), steps, dtype=int)
        # n = n[:np.where(n > self.t.shape)[0][0]]  # make very sure to not overflow time array
        
        # t = self.t[n]*1e-15
        t = self.t
        t = t[t_start:]
        step = (t.shape[0] ) // steps
        t = t[::step]*1e-15
        
        MSD_in = MSD_in[t_start:]
        MSD_in = MSD_in[::step]
        
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
    
    def water_shape(self, force_rdf=True) -> Tuple[float, float]:
        """Retrieve the shape of the average water molecule from the rdf

        Args:
            force_rdf (bool, optional): Switch between force rdf or traditional rdf. Defaults to True=force rdf.

        Returns:
            Tuple:
            r_HO (float): average distance between oxygen and hydrogen in water molecule in Angstrom
            angle (float): average angle HOH /_  in degrees
        """
        if force_rdf is True:
            r_HO = self.rdf_F_r[self.rdf_F_HH2O[0, :].argmax()]
            r_HH = self.rdf_F_r[self.rdf_F_HH[0, :].argmax()]
        else:
            r_HO = self.rdf_r[self.rdf_HH2O[0, :].argmax()]
            r_HH = self.rdf_r[self.rdf_HH[0, :].argmax()]
        # Calculate the cosine of the angle
        cos_gamma = 1 - (r_HH**2) / (2 * r_HO**2)
        
        # Calculate the angle in radians
        gamma_rad = np.arccos(cos_gamma)
        
        # Convert the angle to degrees
        gamma_deg = np.degrees(gamma_rad)
        return r_HO, gamma_deg
        

# folder = r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/1ns/"

# post = Prot_Post(folder)