"""
Created on Wed May 17 11:00:41 2023

@author: Jelle
"""
import Class_diff_hopping_hdf5 as hop
import numpy as np

###############################################################################

###############################################################################

path = ['../RPBE_Production/AIMD/10ps/',
        '../RPBE_Production/MLMD/100ps_2/',
        '../RPBE_Production/MLMD/100ps_Exp_Density/',
        '../RPBE_Production/MLMD/10ns/']

folder = ['i_1']  # , 'i_2'] #, 'i_3', 'i_4', 'i_5']
n_KOH = 1
n_H2O = 110

for i in range(len(folder)):
    Traj = hop.Prot_Hop(path[0]+folder[i], dt=5*1e-16)
    reaction_rate, index, loc_OH = Traj.track_OH(rdf=[32, 2, 5])
    loc_K = Traj.track_K()
    loc_H2O = Traj.track_H2O(index)
    r = Traj.r
    g_OO = Traj.g_OO
    g_HO = Traj.g_HO
    g_KO = Traj.g_KO

    
    np.savez_compressed(path[0]+folder[i]+'/traj', index=index, loc_OH=loc_OH, loc_K=loc_K,
                        loc_H2O=loc_H2O, r=r, g_OO=g_OO, g_HO=g_HO, g_KO=g_KO,
                        N_OH=Traj.N_OH, N_H2O=Traj.N_H2O, N_H3O=Traj.N_H3O)
