# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:30:26 2023

@author: Jelle
"""

import scipy.constants as co
import numpy as np

rho = 1335  # kg/m^3

n_H2O = 55
n_KOH = 1

u_H2O = 2*1.00797 + 1*15.9994
u_KOH = 1*1.00797 + 1*15.9994 + 1*39.0983

m1 = (n_H2O*u_H2O + n_KOH*u_KOH)*co.atomic_mass

L1 = 1e10*np.power(m1/rho, 1/3)
print('box size is', L1, 'Angstrom')

# r = (n_H2O + n_KOH)*rho/(m*co.N_A*1e3)
# print('fftool -r is', r, 'mol/L')

#L2 = 10.9202145373576691*1e-10
#V = np.power(L2, 3)
#m2 = 111*1.00797 + 56*15.9994 + 1*39.0983

#D = (m2/V)*(co.atomic_mass)
#print(D)