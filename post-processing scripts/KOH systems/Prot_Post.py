import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

class Prot_Post:
    def __init__(self, folder):
        self.folder = folder
        self.load_properties()
        self.infer_properties()

    def load_properties(self):
        input = h5py.File(os.path.normpath(self.folder+'output.h5'), 'r')
        
        # load rdfs
        self.rdf_r = input["rdf/r"][()]
        self.rdf_H2OH2O = input["rdf/g_H2OH2O(r)"][()]
        self.rdf_OHH2O = input["rdf/g_OHH2O(r)"][()]
        self.rdf_KH2O = input["rdf/g_KH2O(r)"][()]
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
        
        # Retrieving the msd properties
        self.msd_OH = input["msd/OH"][()]
        self.msd_H2O = input["msd/H2O"][()]
        self.msd_K = input["msd/K"][()]
        
        # retrieving the transient properties
        self.t = input["transient/time"][()]
        self.stress = input["transient/stresses"][()]
        self.energy = input["transient/energies"][()]
        input.close()

    def infer_properties(self):
        # TODO I need to retrieve number of atoms/type ect. here.
        a = 1