# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:01:49 2023

@author: Jelle
"""

import os
from ase.io import read, write

path = '../RPBE_Production/MLMD/100ps_Exp_Density/'
# path = '../RPBE_Production/6m/AIMD/'
folder = ['i_1']

for i in range(len(folder)):
    pwd = path + folder[i] + '/XDATCAR'
    configs = read(pwd, format='vasp-xdatcar', index=':')
    pwd = path + folder[i] + '/trajectory_unwrapped.xyz'
    write(pwd, configs, format='xyz')

    for j in range(len(configs)):
        configs[j].positions = configs[j].positions % configs[0].cell[0, 0]
    pwd = path + folder[i] + '/trajectory_wrapped.xyz'
    write(pwd, configs, format='xyz')