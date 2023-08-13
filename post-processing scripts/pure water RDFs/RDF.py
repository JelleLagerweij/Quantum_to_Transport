# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:29:54 2023

@author: Jelle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
from py4vasp import Calculation
import scipy.optimize as opt
import uncertainties as unc
from uncertainties import unumpy


###############################################################################
# Setting the default figure properties for my thesis
plt.close('all')
plt.rcParams["figure.figsize"] = [6, 5]
label_spacing = 1.1
marker = ['o', 'x', '^', '>']

# Fonts
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.grid"] = "False"

# Sizes of specific parts
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.pad'] = 7
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['legend.fontsize'] = 'large'
# File properties and location
# Done

###############################################################################

# Creating the RDF and read HDF5 functions
def rdf_with_error(path, boodstrap, interpol=False):
    file = [0]*len(boodstrap)
    for i in range(len(boodstrap)):
        file[i] = path + '/' + boodstrap[i] + '/' + 'PCDAT'

    with open(file[0]) as fp:
        n = int(fp.readlines()[6])
    with open(file[0]) as fp:
        dr = float(fp.readlines()[8])

    # Creating the distance indicator (r axis)
    r = np.arange(n*dr, step=dr)
    if interpol is False:
        r_small = r
    else:
        r_small = np.arange(n*dr, step=dr/interpol)

    # Create empty arrays for the g(r)
    stored_HH = np.zeros((len(r_small), len(file)))
    stored_HO = np.zeros((len(r_small), len(file)))
    stored_OO = np.zeros((len(r_small), len(file)))
    stored_N_cord = np.zeros(len(file))
    # Main loop
    for i in range(len(file)):
        RDF_data = pd.read_table(file[i], header=None, delim_whitespace=True,
                                 skiprows=12).to_numpy()
        g_HH = RDF_data[:, 1]
        g_HO = RDF_data[:, 2]
        g_OO = RDF_data[:, 3]

        if interpol is False:
            stored_HH[:, i] = g_HH
            stored_HO[:, i] = g_HO
            stored_OO[:, i] = g_OO
        else:
            g_HH = sp.interpolate.CubicSpline(r, g_HH)
            g_HO = sp.interpolate.CubicSpline(r, g_HO)
            g_OO = sp.interpolate.CubicSpline(r, g_OO)

            stored_HH[:, i] = g_HH(r_small)
            stored_HO[:, i] = g_HO(r_small)
            stored_OO[:, i] = g_OO(r_small)

        g_OO = stored_OO[:, i]
        stop_i = np.argmin(abs(r_small - 3.4*1e-10))
        start_i = np.argmin(abs(r_small - 2.3*1e-10))
        N_cord = np.trapz(g_OO[start_i:stop_i]*(r_small[start_i:stop_i]*1e10)**2,
                          r_small[start_i:stop_i]*1e10)
        stored_N_cord[i] = N_cord*4*np.pi*(64/(12.4**3))

    # Computing typical angle and bond length
    r_HH = r_small[np.argmax(stored_HH, axis=0)]
    r_HO = r_small[np.argmax(stored_HO, axis=0)]
    angle = np.rad2deg(np.arcsin(r_HH/(2*r_HO)))*2

    r_HO = np.array([np.mean(r_HO), np.std([r_HO])])
    angle = np.array([np.mean(angle), np.std([angle])])

    # Computing coordination number water
    N_cord = np.array([np.mean(stored_N_cord), np.std(stored_N_cord)])

    HH = np.zeros((2, len(r_small)))
    HO = np.zeros((2, len(r_small)))
    OO = np.zeros((2, len(r_small)))

    HH[0, :] = np.mean(stored_HH, 1)
    HO[0, :] = np.mean(stored_HO, 1)
    OO[0, :] = np.mean(stored_OO, 1)

    HH[1, :] = np.std(stored_HH, 1)
    HO[1, :] = np.std(stored_HO, 1)
    OO[1, :] = np.std(stored_OO, 1)
    return r_small, HH, HO, OO, angle, r_HO, N_cord


def properties(folder, filter_width=0, skip=1):
    """
    Calculate the mean pressure of the simulation.

    Optionally the results are shown in a graphand with filter.

    Parameters
    ----------
    plotting : Boolean, optional
        Optional plots the pressure as function of time. The default is
        False.
    filter_width : Integer, optional
        Filter width of the moving average filter. The default is 0, which
        results in no filtering.

    Returns
    -------
    pressure : Array of 2 Floats
        The average presure and error of mean of the simulation run in bar.

    """
    df = Calculation.from_path(folder)
    stress = df.stress[:].to_dict()['stress']
    energy  = df.energy[:].to_dict()

    press = np.mean(np.diagonal(stress, axis1=1, axis2=2), axis=1)*1e3
    pressure = statistics(press[0::skip])
    
    e_n = np.array(['total energy   ETOTAL',
                    'nose kinetic   EPS',
                    'nose potential ES',
                    'kinetic energy EKIN'])
    ener = energy[e_n[3]]
    e_kin = statistics(ener[0::skip])
    
    ener = (energy[e_n[0]] - energy[e_n[1]] - energy[e_n[2]] - energy[e_n[3]])
    e_pot = statistics(ener[0::skip])
    
    e_tot = e_kin + e_pot
    
    temp = energy['temperature    TEIN']
    temp = statistics(temp[0::skip])
    return temp, pressure, e_tot, e_kin, e_pot


def statistics(s):
    """
    Calculates the mean value and the error of this estimation.
    The error is compensated for the autocorrolation.

    Parameters
    ----------
    s : Array
        The data measured over time.

    Returns
    -------
    mean : float
        The mean value of the array.
    error : float
        the error estimation, expresed as standard deviation, compensated
        for autocorrolation.

    """
    # collect most important characterstics of array
    N = s.shape[0]  # size
    mean = s.mean()  # mean value
    var = np.var(s)  # variance of entire set
    # If no statistic behaviour exists
    if var == 0.0:
        mean, error, tao, g = mean, 0, 0, 0
    # Otherwise calculate the correct error estimation
    else:
        sp = s - mean  # deviation of the mean per index

        # Calculating the total corrolation
        corr = np.zeros(N)
        corr[0] = 1
        for n in range(1, N):
            corr[n] = np.sum(sp[n:]*sp[:-n]/(var*N))

        # To not fitt too long of a data set, the first time that the
        # corrolation drops under 0.1 is recorded and only the data before
        # that is fitted
        g = np.argmax(corr < 0.1)
        t = np.arange(2*g)
        tao = opt.curve_fit(lambda t, b: np.exp(-t/b),  t,
                            corr[:2*g], p0=(g))[0][0]
        error = np.sqrt(2*tao*s.var()/N)
    return np.array([mean, error])

folder = ['i_1', 'i_2' , 'i_3', 'i_4', 'i_5']
methods = ['RPBE_AIMD', 'RPBE_MLMD', 'r2SCAN_AIMD', 'r2SCAN_MLMD']


for i in range(len(methods)):
    t = np.zeros((len(folder), 2))
    p = np.zeros((len(folder), 2))
    e_tot = np.zeros((len(folder), 2))
    e_kin = np.zeros((len(folder), 2))
    e_pot = np.zeros((len(folder), 2))
    for j in range(len(folder)):
        prop = properties(methods[i] + '/' + folder[j])
        t[j, :], p[j, :], e_tot[j, :], e_kin[j, :], e_pot[j, :] = prop
    t = np.sum(unumpy.uarray(t[:, 0], t[:, 1]))/5
    p = np.sum(unumpy.uarray(p[:, 0], p[:, 1]))/5
    e_tot = np.sum(unumpy.uarray(e_tot[:, 0], e_tot[:, 1]))/5
    e_kin = np.sum(unumpy.uarray(e_kin[:, 0], e_kin[:, 1]))/5
    e_pot = np.sum(unumpy.uarray(e_pot[:, 0], e_pot[:, 1]))/5
    print(methods[i])
    print('temp =', t, 'K')
    print('press =', p, 'bar')
    print('energies =', e_tot, e_kin, e_pot, 'eV')
    
angle_stored = np.zeros((len(methods), 2))
r_HO_stored = np.zeros((len(methods), 2))
N_cord_stored = np.zeros((len(methods), 2))
for i in range(len(methods)):
    r, HH, HO, OO, angle, r_HO, N_cord = rdf_with_error(methods[i], folder,
                                                        interpol=999)
    if i == 0:
        HH_m = np.zeros((len(methods), len(r)))
        HH_std = np.zeros((len(methods), len(r)))
        HO_m = np.zeros((len(methods), len(r)))
        HO_std = np.zeros((len(methods), len(r)))
        OO_m = np.zeros((len(methods), len(r)))
        OO_std = np.zeros((len(methods), len(r)))
    HH_m[i, :] = HH[0, :]
    HH_std[i, :] = HH[1, :]
    HO_m[i, :] = HO[0, :]
    HO_std[i, :] = HO[1, :]
    OO_m[i, :] = OO[0, :]
    OO_std[i, :] = OO[1, :]

    angle_stored[i, :] = angle
    r_HO_stored[i, :] = r_HO
    N_cord_stored[i, :] = N_cord

# ##############################################################################
# Plotting the RDF's for Hydrogen-Hydrogen interactions
# Loading Source and Experimental rdfs
rdf_rpbe = pd.read_csv("HH_rpbe.csv", header=None, sep=';',
                      decimal=",")
rdf_rpbe = sp.interpolate.CubicSpline(rdf_rpbe[0]*1e-10, rdf_rpbe[1])
rdf_rpbe = rdf_rpbe(r)

rdf_exp = pd.read_csv("HH_exp.csv", header=None, sep=';',
                      decimal=",")
rdf_exp = sp.interpolate.CubicSpline(rdf_exp[0]*1e-10, rdf_exp[1])
rdf_exp = rdf_exp(r)

plt.figure('HH_RPBE')
plt.plot(r*1e10, HH_m[0, :], label='RPBE-D3 AIMD')
plt.fill_between(r*1e10, HH_m[0, :] - HH_std[0, :], y2= HH_m[0, :] + HH_std[0, :], alpha=0.3)
plt.plot(r*1e10, HH_m[1, :], label='RPBE-D3 MLMD')
plt.fill_between(r*1e10, HH_m[1, :] - HH_std[1, :], y2= HH_m[1, :] + HH_std[1, :], alpha=0.3)
plt.plot(r*1e10, rdf_exp, label='Experimental')
plt.plot(r*1e10, rdf_rpbe, label='RPBE-D3 Reference')
plt.xlabel('r/[Angstrom]')
plt.ylabel('pair-correlation function/[1/Angstrom]')
plt.legend(labelspacing=label_spacing, frameon=False)
plt.xlim(1, 7)
plt.ylim(0, 2.5)
# plt.grid()
# plt.savefig(figures+'\HH_RPBE.svg')

plt.figure('HH_r2SCAN')
plt.plot(r*1e10, HH_m[2, :], label='rVV10-r2SCAN AIMD')
plt.fill_between(r*1e10, HH_m[2, :] - HH_std[2, :], y2= HH_m[2, :] + HH_std[2, :], alpha=0.3)
plt.plot(r*1e10, HH_m[3, :], label='rVV10-r2SCAN MLMD')
plt.fill_between(r*1e10, HH_m[3, :] - HH_std[3, :], y2= HH_m[3, :] + HH_std[3, :], alpha=0.3)
plt.plot(r*1e10, rdf_exp, label='experimental')
plt.xlabel('r/[Angstrom]')
plt.ylabel('pair-correlation function/[1/Angstrom]')
plt.legend(labelspacing=label_spacing, frameon=False)
plt.xlim(1, 7)
plt.ylim(0, 2.5)
# plt.grid()
# plt.savefig(figures+'\HH_r2SCAN.svg')

##############################################################################
# Plotting the RDF's for Hydrogen-Oxygen interactions
# Loading Source and Experimental rdfs
rdf_rpbe = pd.read_csv("OH_rpbe.csv", header=None, sep=';',
                      decimal=",")
rdf_rpbe = sp.interpolate.CubicSpline(rdf_rpbe[0]*1e-10, rdf_rpbe[1])
rdf_rpbe = rdf_rpbe(r)

rdf_exp = pd.read_csv("OH_exp.csv", header=None, sep=';',
                      decimal=",")
rdf_exp = sp.interpolate.CubicSpline(rdf_exp[0]*1e-10, rdf_exp[1])
rdf_exp = rdf_exp(r)


plt.figure('OH_RPBE')
plt.plot(r*1e10, HO_m[0, :], label='RPBE-D3 AIMD')
plt.fill_between(r*1e10, HO_m[0, :] - HO_std[0, :], y2= HO_m[0, :] + HO_std[0, :], alpha=0.3)
plt.plot(r*1e10, HO_m[1, :], label='RPBE-D3 MLMD')
plt.fill_between(r*1e10, HO_m[1, :] - HO_std[1, :], y2= HO_m[1, :] + HO_std[1, :], alpha=0.3)
plt.plot(r*1e10, rdf_exp, label='Experimental')
plt.plot(r*1e10, rdf_rpbe, label='RPBE-D3 Reference')
plt.xlabel('r/[Angstrom]')
plt.ylabel('pair-correlation function/[1/Angstrom]')
plt.legend(labelspacing=label_spacing, frameon=False)
plt.xlim(1.5, 7)
plt.ylim(0, 2.0)
# plt.grid()
# plt.savefig(figures+'\HO_RPBE.svg')

plt.figure('HO_r2SCAN')
plt.plot(r*1e10, HO_m[2, :], label='rVV10-r2SCAN AIMD')
plt.fill_between(r*1e10, HO_m[2, :] - HO_std[2, :], y2= HO_m[2, :] + HO_std[2, :], alpha=0.3)
plt.plot(r*1e10, HO_m[3, :], label='rVV10-r2SCAN MLMD')
plt.fill_between(r*1e10, HO_m[3, :] - HO_std[3, :], y2= HO_m[3, :] + HO_std[3, :], alpha=0.3)
plt.plot(r*1e10, rdf_exp, label='experimental')
plt.xlabel('r/[Angstrom]')
plt.ylabel('pair-correlation function/[1/Angstrom]')
plt.legend(labelspacing=label_spacing, frameon=False)
plt.xlim(1.5, 7)
plt.ylim(0, 2.0)
# plt.grid()
# plt.savefig(figures+'\HO_r2SCAN.svg')

# ##############################################################################
# # Plotting the RDF's for Hydrogen-Oxygen interactions
# # Loading Source and Experimental rdfs
rdf_rpbe = pd.read_csv("OO_rpbe.csv", header=None, sep=';',
                      decimal=",")
rdf_rpbe = sp.interpolate.CubicSpline(rdf_rpbe[0]*1e-10, rdf_rpbe[1])
rdf_rpbe = rdf_rpbe(r)

rdf_exp = pd.read_csv("OO_exp.csv", header=None, sep=';',
                      decimal=",")
rdf_exp = sp.interpolate.CubicSpline(rdf_exp[0]*1e-10, rdf_exp[1])
rdf_exp = rdf_exp(r)

# stop_i = start + np.argmin(rdf_exp[start:stop])
stop_i = np.argmin(abs(r - 3.4*1e-10))
start_i = np.argmin(abs(r - 2.3*1e-10))
start_i = 0
N_cord = np.trapz(rdf_exp[start_i:stop_i]*(r[start_i:stop_i,]*1e10)**2, r[start_i:stop_i]*1e10)*4*np.pi*(64/(12.4**3))
print('experimental Ncord =', N_cord)

N_cord = np.trapz(rdf_rpbe[:stop_i]*(r[:stop_i,]*1e10)**2, r[:stop_i]*1e10)*4*np.pi*(64/(12.4**3))
print('reference Ncord =', N_cord) 


plt.figure('OO_RPBE')
plt.plot(r*1e10, OO_m[0, :], label='RPBE-D3 AIMD')
plt.fill_between(r*1e10, OO_m[0, :] - OO_std[0, :], y2= OO_m[0, :] + OO_std[0, :], alpha=0.3)
plt.plot(r*1e10, OO_m[1, :], label='RPBE-D3 MLMD')
plt.fill_between(r*1e10, OO_m[1, :] - OO_std[1, :], y2= OO_m[1, :] + OO_std[1, :], alpha=0.3)
plt.plot(r*1e10, rdf_exp, label='Experimental')
plt.plot(r*1e10, rdf_rpbe, label='RPBE-D3 Reference')
plt.xlabel('r/[Angstrom]')
plt.ylabel('pair-correlation function/[1/Angstrom]')
plt.legend(labelspacing=label_spacing, frameon=False)
plt.xlim(2, 7)
plt.ylim(0, 3.5)
plt.grid()
# plt.savefig(figures+'\OO_RPBE.svg')

plt.figure('OO_r2SCAN')
plt.plot(r*1e10, OO_m[2, :], label='rVV10-r2SCAN AIMD')
plt.fill_between(r*1e10, OO_m[2, :] - OO_std[2, :], y2= OO_m[2, :] + OO_std[2, :], alpha=0.3)
plt.plot(r*1e10, OO_m[3, :], label='rVV10-r2SCAN MLMD')
plt.fill_between(r*1e10, OO_m[3, :] - OO_std[3, :], y2= OO_m[3, :] + OO_std[3, :], alpha=0.3)
plt.plot(r*1e10, rdf_exp, label='experimental')
plt.xlabel('r/[Angstrom]')
plt.ylabel('pair-correlation function/[1/Angstrom]')
plt.legend(labelspacing=label_spacing, frameon=False)
plt.xlim(2, 7)
plt.ylim(0, 3.5)
plt.grid()
# plt.savefig(figures+'\OO_r2SCAN.svg')

# ###############################################################################
# Calculating bond length and angles by hand from output data