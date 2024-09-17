from ase.io import read, write
from ase import Atoms
import numpy as np

path = "post-processing scripts/KOH systems/test_output/1ns/"
# Open the trajectory file in read mode
traj = read(path+"traj_processed_wrapped.xyz", index='200:')
pos_compare = -1

L = 14.9247099999999993   # boxsize in

N_F = 1
N_O = 15
N_H = N_O*2+N_F

H_index = np.zeros(N_H, dtype=int)
# get closest H and O at first timestep
pos_F = traj[pos_compare].positions[0]
pos_O = traj[pos_compare].positions[1:111]
pos_H = traj[pos_compare].positions[111:]

# Fluor Oxygen
# cartesian distances with pbc
d_FO = (pos_O - pos_F + L/2)%L -L/2
# radial distances squared work to find closest ones
r2_FO = np.sum(d_FO**2, axis=1)
# sort and filter
O_index = np.argsort(r2_FO)[:N_O]
pos_O_filtered = pos_O[O_index]

# Fluor Hydrogen
d_FH = (pos_H - pos_F + L/2)%L -L/2
r2_FH = np.sum(d_FH**2, axis=1)
H_index[0] = np.argsort(r2_FH)[0]

# Oxygen Hydrogen, only use clossest
for i, pos_Oi in enumerate(pos_O_filtered):
    d_FH = (pos_H - pos_Oi + L/2)%L -L/2
    r2_FH = np.sum(d_FH**2, axis=1)
    H_index[1+(2*i):1+(2*(i+1))] = np.argsort(r2_FH)[:2]

save = []
for snapshot in traj:
    pos_F = snapshot.positions[0]
    pos_O = snapshot.positions[1:111]
    pos_H = snapshot.positions[111:]
    
    pos_O_filtered = pos_O[O_index]
    pos_H_filtered = pos_H[H_index]
        
    # Apply PBC to make sure all positions are the closest image compared to fluorine
    pos_O_filtered = (pos_O_filtered - pos_F + L / 2) % L - L / 2 + pos_F
    pos_H_filtered = (pos_H_filtered - pos_F + L / 2) % L - L / 2 + pos_F

    center_of_box = np.array([L / 2, L / 2, L / 2])
    shift_vector = center_of_box - traj[pos_compare].positions[0]  # only shift compared to first snapshot
    
    filtered_positions = (np.vstack([pos_F, pos_O_filtered, pos_H_filtered]) + shift_vector)%L
    symbols = ['F']*N_F + ['O']*N_O + ['H']*N_H
    
    filtered_snapshot = Atoms(symbols=symbols, positions=filtered_positions, cell=[L, L, L], pbc=True)
    save.append(filtered_snapshot)

write(path+"traj_processed.pdb", save, format='proteindatabank')
