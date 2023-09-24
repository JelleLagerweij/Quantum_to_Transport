import numpy as np
import matplotlib.pyplot as plt

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_lengths
    
a = np.array([1, 1, 1, 3, 3, 1, 4, 4, 4, 4, 4, 4, 3, 3, 2, 2])

run_lengths = find_runs(a)

hist_length, bin_edges = np.histogram(run_lengths, bins=200, range=(0, 2000), density=True)

plt.plot(bin_edges[:-1], hist_length)