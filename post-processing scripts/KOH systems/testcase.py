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
        r"\usepackage[version=3]{mhchem}"]),
    'font.size': 18}

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)
# plt.style.use('default')

plt.rcParams["axes.grid"] = "False"

# Sizes of specific parts
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize']= 0.75
plt.rcParams['xtick.major.pad'] = 7
plt.rcParams['ytick.major.pad'] = 7

plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams['ytick.right'] = True

plt.rcParams["legend.frameon"] = False
# plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelpad']= 7
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 15

# # figures = r'C:\Users\Jelle\Delft University of Technology\Jelle Lagerweij Master - Documents\General\Personal Thesis files\01 Defence Presentation\Figures'
figures = r'C:\Users\Jelle\Documents\TU jaar 6\Project KOH(aq)\Progress_meeting_3\figures'
# plt.rcParams['savefig.directory'] = figures
# Done

###############################################################################

###############################################################################

path = ['../../../RPBE_Production/AIMD/10ps/',
        '../../../RPBE_Production/MLMD/100ps_2/',
        '../../../RPBE_Production/MLMD/100ps_Exp_Density/']
path_short = [r'AIMD \SI{10}{\ps}', r'MLMD \SI{100}{\ps}',
              r'AIMD \SI{100}{\ps} $\rho_\text{exp}$']
folder = ['i_1', 'i_2', 'i_3', 'i_4', 'i_5']

n_bins = 1000
length = 1000

for j in range(len(path)):
    hists_s = np.zeros(n_bins)
    for i in range(len(folder)):
        Traj = hop.Prot_Hop(path[j]+folder[i])
        index, loc_OH, loc_K, loc_H2O = Traj.loading()  # load postprocessed trajectory
        hist, bins = Traj.react_time(plotting=False, n_bins=n_bins, range=length)
        hists_s += hist/len(folder)

    plt.figure('reaction_spacing')
    plt.plot(bins, hists_s, label=path_short[j])

plt.figure('reaction_spacing')
#plt.plot(bins, hists_s, label='average')
# plt.xlabel(r'$t$/[\si{\ps}]')
# plt.ylabel(r'probability/[-]')
plt.xlabel('time')
plt.legend()
plt.savefig(figures + '/reaction_spacing')