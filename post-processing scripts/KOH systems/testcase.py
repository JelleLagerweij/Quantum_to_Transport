import numpy as np
from ase import Atoms

# Example position arrays for multiple timesteps
positions_combined = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                               [[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1]],
                               [[0.2, 0.2, 0.2], [1.2, 1.2, 1.2], [2.2, 2.2, 2.2]]])

# Concatenate position arrays for all timesteps along the first axis
# positions = np.concatenate(positions_combined, axis=0)

# Example atomic symbols
symbols = ['H', 'H', 'H']

# Create an ASE Atoms object directly from combined position array
atoms = Atoms(symbols, positions=positions_combined)

# Now 'atoms' contains Atoms objects for all timesteps
