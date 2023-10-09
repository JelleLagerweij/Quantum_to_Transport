"""
Created on Wed May 17 11:00:41 2023

@author: Jelle
"""
import Class_diff_hopping_hdf5 as hop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.constants as co
import uncertainties as unc
from uncertainties import unumpy
plt.close('all')

###############################################################################
# Setting the default figure properties for all documents
plt.close('all')
plt.rcParams["figure.figsize"] = [8, 6]
label_spacing = 1.1
marker = ['o', 'x', '^', '>']

# Papers
# # Fonts
# plt.rcParams["svg.fonttype"] = "none"
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["axes.grid"] = "False"

# # Sizes of specific parts
# plt.rcParams['axes.labelsize'] = 'large'
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['xtick.major.pad'] = 7
# plt.rcParams['ytick.major.pad'] = 5
# plt.rcParams['xtick.labelsize'] = 'large'
# plt.rcParams['ytick.labelsize'] = 'large'
# plt.rcParams['lines.markersize'] = 10
# plt.rcParams['lines.markeredgewidth'] = 2
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['xtick.major.size'] = 5
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['ytick.major.size'] = 5
# plt.rcParams['ytick.major.width'] = 2
# plt.rcParams['legend.fontsize'] = 'large'
# plt.rcParams['legend.frameon'] = False
# plt.rcParams['legend.labelspacing'] = 0.75
# plt.rcParams['axes.grid'] = True
# # File properties and location
# # Done

# Presentations
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "lualatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": 'serif',
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\RequirePackage{amsmath}",
        r"\RequirePackage{fontspec}",   # unicode math setup
        r"\setmainfont[Scale = MatchLowercase]{DejaVu Serif}",
        r"\setsansfont[Scale = MatchLowercase]{DejaVu Sans}",
        r"\setmonofont[Scale = MatchLowercase]{DejaVu Sans Mono}",
        r"\usepackage{unicode-math}",
        r"\setmathfont[Scale = MatchLowercase]{DejaVu Math TeX Gyre}", 
        r"\usepackage{siunitx}",
        r"\usepackage[version=3]{mhchem}"])}

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)
# plt.style.use('default')

plt.rcParams["axes.grid"] = "False"

# Sizes of specific parts
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelpad']= 7
plt.rcParams['xtick.major.pad'] = 4
plt.rcParams['ytick.major.pad'] = 4

plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams['ytick.right'] = True

plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize']= 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# # figures = r'C:\Users\Jelle\Delft University of Technology\Jelle Lagerweij Master - Documents\General\Personal Thesis files\01 Defence Presentation\Figures'
figures = r'C:\Users\Jelle\Documents\TU jaar 6\Project KOH(aq)\Progress_meeting_4\figures'
# plt.rcParams['savefig.directory'] = figures
# Done

###############################################################################

###############################################################################

path = ['../../../RPBE_Production/AIMD/10ps/',
        '../../../RPBE_Production/MLMD/100ps_2/',
        '../../../RPBE_Production/MLMD/100ps_Exp_Density/',
        '../../../RPBE_Production/MLMD/10ns/']

folder = ['i_1'] #, 'i_2', 'i_3', 'i_4', 'i_5']
path = '../../../RPBE_Production/MLMD/100ps_Exp_Density/'
# folder = ['i_1']

n_KOH = 1
n_H2O = 55

visc = np.zeros((len(folder), 2))
press = np.zeros((len(folder), 2))
e_kin = np.zeros((len(folder), 2))
e_pot = np.zeros((len(folder), 2))
e_tot = np.zeros((len(folder), 2))
Temp = np.zeros((len(folder), 2))
diff_OH = np.zeros(len(folder))
diff_K = np.zeros(len(folder))
diff_H2O = np.zeros(len(folder))
OO = np.zeros((len(folder), 1984))
HO = np.zeros((len(folder), 1984))
KO = np.zeros((len(folder), 1984))
d_OO = np.zeros(len(folder))
d_HO = np.zeros(len(folder))
d_KO = np.zeros(len(folder))
n_OO = np.zeros(len(folder))
n_HO = np.zeros(len(folder))
n_KO = np.zeros(len(folder))
hists_s = np.zeros(100)

for i in range(len(folder)):
    Traj = hop.Prot_Hop(path+folder[i])
    index, loc_OH, loc_K, loc_H2O = Traj.loading()  # load postprocessed trajectory
    # hist, bins = Traj.react_time(plotting=True, n_bins=100, range=500)

    # hists_s += hist/len(folder)
    

    # visc[i, :] = Traj.viscosity(cubicspline=10, plotting=True, padding=0)

    # r, rdfs, n_conf = Traj.rdf(interpol=64, plotting=False,
    #                            r_end=[4.2, 3.3, 3.5])
    # n_OO[i], n_HO[i], n_KO[i] = n_conf
    # OO[i, :], HO[i, :], KO[i, :] = rdfs

    # d_OO[i] = r[np.argmax(OO[i, :])]
    # d_HO[i] = r[np.argmax(HO[i, :])]
    # d_KO[i] = r[np.argmax(KO[i, :])]

    # # sanity checks
    # # ave_beef, ave_bees = Traj.bayes_error(plotting=True, move_ave=2500)
    # press[i, :] = Traj.pressure(plotting=True, filter_width=2500, skip=100)
    # e_kin[i, :] = Traj.kin_energy(plotting=True, filter_width=2500, skip=100)
    # e_pot[i, :] = Traj.pot_energy(plotting=True, filter_width=2500, skip=100)
    # e_tot[i, :] = Traj.tot_energy(plotting=True, filter_width=2500, skip=100)
    # Temp[i, :] = Traj.temperature(plotting=True, filter_width=2500, skip=100)

    # msdOH = Traj.windowed_MSD(loc_OH, n_KOH)
    # msdK = Traj.windowed_MSD(loc_K, n_KOH)
    # msdH2O = Traj.windowed_MSD(loc_H2O, n_H2O)

    # # t = np.arange(len(Traj.t))

    # multiple window loglog
    # plt.figure('multiple window loglog OH')
    # plt.loglog(Traj.t[1:]*1e12, msdOH[1:], label=folder[i])

    # # multiple window loglog
    # plt.figure('multiple window loglog K')
    # plt.loglog(Traj.t[1:]*1e12, msdK[1:], label=folder[i])

    # # multiple window loglog
    # plt.figure('multiple window loglog H2O')
    # plt.loglog(Traj.t[1:]*1e12, msdH2O[1:], label=folder[i])

    plt.figure('OH index')
    plt.plot(Traj.t*1e12, index - Traj.N_H, label=folder[i])

    # n = np.arange(start=1000, stop=len(Traj.t), step=100)
    # n = np.arange(start=10000, stop=len(Traj.t), step=100)
    # t_test = Traj.t[n]
    # MSD_test = msdOH[n]
    # diff_OH[i] = Traj.diffusion(MSD_test, n_KOH, t=t_test, plotting=True)
    # MSD_test = msdK[n]
    # diff_K[i] = Traj.diffusion(MSD_test, n_KOH, t=t_test, plotting=True)
    # MSD_test = msdH2O[n]
    # diff_H2O[i] = Traj.diffusion(MSD_test, n_H2O, t=t_test, plotting=True)

    # plt.figure('Number OH-')
    # plt.plot(Traj.t*1e12, Traj.N_OH, label=folder[i])

    # plt.figure('Number H3O+')
    # plt.plot(Traj.t*1e12, Traj.N_H3O, label=folder[i])

    # plt.figure('Number H2O')
    # plt.plot(Traj.t*1e12, Traj.N_H2O, label=folder[i])

#     plt.figure('xyz OH-')
#     plt.plot(Traj.t*1e12, Traj.O_loc_stored[:, :, 0], label='x')
#     plt.plot(Traj.t*1e12, Traj.O_loc_stored[:, :, 1], label='y')
#     plt.plot(Traj.t*1e12, Traj.O_loc_stored[:, :, 2], label='z')

#     plt.figure('stepsize')
#     step = Traj.O_loc_stored[1:, :, :] - Traj.O_loc_stored[:-1, :, :]
#     dis = np.linalg.norm(step, axis=2)
#     plt.plot(Traj.t[:-1]*1e12, dis, label=folder[i])


# # rdfs= np.array([r, np.mean(OO, axis=0), np.std(OO, axis=0),
# #                 np.mean(HO, axis=0), np.std(HO, axis=0),
# #                 np.mean(KO, axis=0), np.std(KO, axis=0)])
# # np.save(path+'rdfs.npy', rdfs)

t_max = int(len(Traj.t)*1e-3)
# plt.figure('pressure')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$P$/[\si{\bar}]')
# plt.xlim(0, t_max)
# plt.legend()
# plt.savefig(figures + '/pressure')

# plt.figure('kinetic energy')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$E_\text{kin}$/[\si{\eV}]')
# plt.xlim(0, t_max)
# plt.legend()
# plt.savefig(figures + '/kinetic energy')

# plt.figure('potential energy')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$E_\text{pot}$/[\si{\eV}]')
# plt.xlim(0, t_max)
# plt.legend()
# plt.savefig(figures + '/potential energy')

# plt.figure('total energy')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$E_\text{tot}$/[\si{\eV}]')
# plt.xlim(0, t_max)
# plt.legend()
# plt.savefig(figures + '/total energy')

# plt.figure('temperature')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$T$/[\si{\K}]')
# plt.xlim(0, t_max)
# plt.legend()
# plt.savefig(figures + '/temperature')

# # plt.figure('BEEF')
# # plt.xlabel('time in/[ps]')
# # plt.ylabel('force/[eV/angstrom]')
# # plt.xlim(0, t_max)
# # plt.ylim(0, 0.15)
# # plt.legend()
# # plt.savefig(figures + '/BEEF')

# # plt.figure('BEES')
# # plt.xlabel('time in/[ps]')
# # plt.ylabel('stress/[bar]')
# # plt.xlim(0, t_max)
# # plt.legend()
# # plt.savefig(figures + '/BEES')

plt.figure('OH index')
plt.xlabel(r'$t$/[\si{\ps}]')
plt.ylabel(r'Index of oxygen of OH-')
plt.xlim(0, t_max)
plt.ylim(-5, 60)
plt.legend()
plt.savefig(figures + '/OH_idex')

# # multiple window loglog
# plt.figure('multiple window loglog OH')
# plt.legend()
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$MSD_\text{\ce{OH-}}$/[\si{\angstrom\squared}]')
# plt.savefig(figures + '/MSD OH')

# # multiple window loglog
# plt.figure('multiple window loglog K')
# plt.legend()
# plt.xlabel(r'time in/[ps]')
# plt.ylabel(r'$MSD_\text{\ce{K+}}$/[\si{\angstrom\squared}]')
# plt.savefig(figures + '/MSD K')

# # multiple window loglog
# plt.figure('multiple window loglog H2O')
# plt.legend()
# plt.xlabel(r'time in/[ps]')
# plt.ylabel(r'$MSD_\text{\ce{H2O}}$/[\si{\angstrom\squared}]')
# plt.savefig(figures + '/MSD H2O')

# plt.figure('Number OH-')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$N_\text{\ce{OH-}}$')
# plt.xlim(0, t_max)
# plt.ylim(-0.5, 5)
# plt.legend()
# plt.savefig(figures + '/N_OH')

# plt.figure('Number H3O+')
# plt.legend()
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$N_\text{\ce{H3O+}}$')
# plt.xlim(0, t_max)
# plt.ylim(-0.5, 5)
# plt.savefig(figures + '/N_H3O')

# plt.figure('Number H2O')
# plt.legend()
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'$N_\text{\ce{H2O}}$')
# plt.xlim(0, t_max)
# plt.savefig(figures + '/N_H2O')

# plt.figure('xyz OH-')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'position/[\si{\angstrom}]')
# plt.xlim(0, t_max)
# plt.legend()
# plt.savefig(figures + '/trajectory')

# plt.figure('stepsize')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'stepsize/[\si{\angstrom}]')
# plt.xlim(0, t_max)
# plt.legend()
# plt.savefig(figures + '/stepsize')

# plt.figure('reaction_spacing')
# plt.plot(bins, hists_s, label='average')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'probability/[-]')
# plt.legend()
# plt.savefig(figures + '/reaction_spacing')

# p = np.sum(unumpy.uarray(press[:, 0], press[:, 1]))/5
# t = np.sum(unumpy.uarray(Temp[:, 0], Temp[:, 1]))/5
# e_tot = np.sum(unumpy.uarray(e_tot[:, 0], e_tot[:, 1]))/5
# e_kin = np.sum(unumpy.uarray(e_kin[:, 0], e_kin[:, 1]))/5
# e_pot = np.sum(unumpy.uarray(e_pot[:, 0], e_pot[:, 1]))/5
# n_oo = unc.ufloat(np.mean(n_OO), np.std(n_OO)/np.sqrt(5))
# n_ho = unc.ufloat(np.mean(n_HO), np.std(n_HO)/np.sqrt(5))
# n_ko = unc.ufloat(np.mean(n_KO), np.std(n_KO)/np.sqrt(5))
# d_oo = unc.ufloat(np.mean(d_OO), np.std(d_OO)/np.sqrt(5))
# d_ho = unc.ufloat(np.mean(d_HO), np.std(d_HO)/np.sqrt(5))
# d_ko = unc.ufloat(np.mean(d_KO), np.std(d_KO)/np.sqrt(5))

# viscosity = unc.ufloat(np.mean(visc[:, 0]),
#                         np.std(visc[:, 1])/np.sqrt(5))
# D_cor = co.k*t*2.837298/(6*np.pi*Traj.L*1e-10*viscosity)
# D_H2O = unc.ufloat(np.mean(diff_H2O), np.std(diff_H2O)/np.sqrt(5)) + D_cor
# D_OH = unc.ufloat(np.mean(diff_OH), np.std(diff_OH)/np.sqrt(5)) + D_cor
# D_K = unc.ufloat(np.mean(diff_K), np.std(diff_K)/np.sqrt(5)) + D_cor
# sigma = (1*co.eV**2)*(D_K + D_OH)/(co.k*t*(Traj.L*1e-10)**3)

# print('first peak in Angstrom =', d_oo)
# print('nOO =', n_oo)
# print('first peak in Angstrom =', d_ko)
# print('nKO =', n_ko)
# print('first peak in Angstrom =', d_ho)
# print('nHO =', n_ho)


# print('E_tot in eV =', e_tot)
# print('E_kin in eV', e_kin)
# print('E_pot in eV', e_pot)
# print('temperature in K =', t)
# print('pressure in bar =', p)

# print('viscosity in mPas (or cP) =', viscosity*1e3)
# print('D_s correction in 10^-9 m^2/s =', D_cor*1e9)
# print('D_s H2O in 10^-9 m^2/s =', D_H2O*1e9)
# print('D_s K in 10^-9 m^2/s =', D_K*1e9)
# print('D_s OH in 10^-9 m^2/s =', D_OH*1e9)
# print('Electric conductivity in S/m', sigma)
