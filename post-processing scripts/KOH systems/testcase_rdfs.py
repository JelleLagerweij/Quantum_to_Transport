import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os

path = r'C:\Users\vlagerweij\Documents\TU jaar 6\Project KOH(aq)\Progress_meeting_17/figures'

load = "/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/combined_simulation/single_core/output.npz"
load = os.path.normpath(load)
loaded = np.load(load)

r = loaded['r_rdf']
rdf_H2OH2O = loaded['rdf_H2OH2O']
rdf_KH2O = loaded['rdf_KH2O']
rdf_OHH2O = loaded['rdf_OHH2O']

rdf_mancinelli = np.array([[0.12749, -0.00058], [0.37092, -0.00058], [0.61435, -0.00058],
                           [0.85777, -0.00058], [1.10120, -0.00058], [1.34463, -0.00058],
                           [1.58806, -0.00058], [1.83148, -0.00058], [2.07491, -0.00058],
                           [2.30263, -0.00045], [2.46575, 0.07954], [2.51370, 0.31026],
                           [2.58219, 0.85671], [2.62329, 1.37583], [2.67808, 1.99211],
                           [2.73973, 2.20158], [2.80137, 2.10747], [2.89041, 1.80389],
                           [2.96575, 1.53066], [3.04110, 1.34244], [3.13500, 1.24658],
                           [3.30238, 1.17096], [3.52762, 1.13962], [3.81407, 1.09849],
                           [4.05928, 1.04909], [4.29492, 0.99549], [4.56651, 0.93906],
                           [4.83488, 0.90981], [5.04074, 0.90822], [5.30229, 0.91826],
                           [5.52386, 0.92144], [5.79138, 0.96152], [6.04080, 1.01770],
                           [6.27824, 1.05450], [6.52728, 1.06444], [6.76315, 1.05641],
                           [7.01377, 1.03596], [7.27523, 1.00858], [7.53414, 0.98555],
                           [7.66981, 0.98091]])

rdf_zhang = np.array([[2.21355, -0.00355], [2.43671, 0.04415], [2.49844, 0.20121], [2.60436, 0.88999],
                      [2.64174, 1.46971], [2.68536, 1.88871], [2.72099, 2.12496], [2.77446, 2.26290],
                      [2.86953, 1.99763], [2.90112, 1.85733], [2.92524, 1.75454], [2.95540, 1.65592],
                      [3.00365, 1.53229], [3.05190, 1.43089], [3.11824, 1.33306], [3.26902, 1.24322],
                      [3.48615, 1.16265], [3.70327, 1.07375], [3.99880, 0.98232], [4.31846, 0.93841],
                      [4.90349, 0.90928], [5.27743, 0.87989], [5.59106, 0.95564], [5.90468, 1.04155],
                      [6.33516, 1.07026], [6.64762, 1.04486], [7.02650, 1.00105], [7.40043, 0.97174],
                      [7.73215, 0.96764]])

rdf_soper = np.array([[2.26689, 0.00000], [2.43605, 0.09669], [2.49248, 0.32416], [2.52293, 0.54030],
                      [2.54891, 0.82825], [2.56758, 1.12442], [2.58991, 1.46153], [2.63360, 1.84942],
                      [2.66293, 2.45931], [2.73859, 2.73043], [2.84950, 2.14197], [2.94513, 1.53789],
                      [3.04966, 1.10205], [3.23295, 0.84644], [3.98439, 1.02139], [4.63663, 1.13768],
                      [5.13032, 0.94686], [5.71760, 0.89316], [6.22492, 0.99334], [6.77574, 1.06118],
                      [7.38147, 0.99939]])

rdf_heuft = np.array([[2.16393, -0.00188], [2.38604, 0.09572], [2.46219, 0.40353], [2.50661, 0.70008],
                      [2.62084, 1.55593], [2.67160, 1.90128], [2.72237, 2.05143], [2.77314, 2.08709],
                      [2.84929, 1.91254], [2.93496, 1.63476], [3.03014, 1.40578], [3.20148, 1.25563],
                      [3.35695, 1.15991], [3.57271, 1.03791], [3.89635, 1.03416], [4.31200, 1.05293],
                      [4.57536, 1.03228], [4.93707, 0.94970], [5.32417, 0.95721], [5.75568, 1.00413],
                      [6.18086, 1.03041], [6.72660, 1.06794]])

def smoothen_rdfs(r, rdf, step):
    r_small = np.arange(start=r[0], stop=r[-1], step=step)
    g = sp.interpolate.CubicSpline(r, rdf)
    g = g(r_small)
    return r_small, g

g_H2OH2O = smoothen_rdfs(r, rdf_H2OH2O, 0.01)[1]
g_KH2O = smoothen_rdfs(r, rdf_KH2O, 0.01)[1]
r, g_OHH2O = smoothen_rdfs(r, rdf_OHH2O, 0.01)

r_m, g_m = smoothen_rdfs(rdf_mancinelli[:, 0], rdf_mancinelli[:, 1], 0.01)
r_z, g_z = smoothen_rdfs(rdf_zhang[:, 0], rdf_zhang[:, 1], 0.01)
r_s, g_s = smoothen_rdfs(rdf_soper[:, 0], rdf_soper[:, 1], 0.01)
r_h, g_h = smoothen_rdfs(rdf_heuft[:, 0], rdf_heuft[:, 1], 0.01)

plt.figure()
plt.plot(r, g_H2OH2O, label='This Work, 110:12 (1:9.2)')
plt.plot(r_h, g_h, label='PhD Thesis Heuft (UvA), 113:12 LiOH (1:9.4)')
plt.plot(r_m, g_m, label='Mancinelli experimental 1:10 KCl(aq)')
plt.plot(r_z, g_z, label='Zhang DFT 1:10 NaCl(aq)') 
plt.plot(r_s, g_s, label='Soper experimental Water pure')
# plt.plot(r, g_KH2O, label='Potassium-Water')
# plt.plot(r, g_OHH2O, label='Hydroxide-Water')
plt.legend()
plt.xlabel('radius in A')
plt.ylabel('g(r) of Water-Water')
plt.xlim(2.3, 7.5)
plt.savefig(path + r'\rdfs_compare')    