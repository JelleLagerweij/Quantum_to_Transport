import numpy as np
import h5py
import os
import pandas as pd

folder = os.path.normpath(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/longest_up_till_now")

loaded = np.load(folder + r"/single_core/output.npz")
output = h5py.File(folder + r"/single_core/output.h5", "w")

rdfs = loaded["r_rdf"]
rdf_H2OH2O = loaded["rdf_H2OH2O"]

dset = output.create_dataset("rdfs/r", data=rdfs)
dset = output.create_dataset("rdfs/g_H2OH2O", data=rdfs)

data = np.hstack((rdfs, rdf_H2OH2O))
df = pd.DataFrame(data, columns=["r", "g_H2OH2O"])

dset2 = output.create_dataset("rdfs", data=df)

output.close()