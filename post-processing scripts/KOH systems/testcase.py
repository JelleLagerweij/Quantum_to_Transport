import numpy as np

# Create an empty zero-length array
N_OH = 3
OH_0 = np.array([1, 2, 3])
OH_i = np.array([1, 2, 3, 4])

# Append items to the array
c = np.setdiff1d(OH_i, OH_0)
print(c)