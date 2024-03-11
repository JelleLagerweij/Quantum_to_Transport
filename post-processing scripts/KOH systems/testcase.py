from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# # Assuming self.rdf is a numpy array with equal sizes on each core
# # Replace this with your actual array
# rdf = (rank + 1)*np.array([1, 2, 3, 4, 5])  # Example array

# # Perform local sum on each process
# local_sum = rdf

# # Use MPI Reduce to sum the arrays across all processes
# global_sum = np.zeros_like(local_sum)
# comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)


# if rank == 0:
#     local_sum = global_sum/float(size)
#     print(f"Process {rank}: Local sum = {local_sum}, Global sum = {global_sum}")
# else:
#     print(f"Process {rank}: Local sum = {local_sum}")

rdf = (rank + 1)*np.array([1, 2, 3, 4, 5])  # Example array

# Use MPI Reduce to sum the arrays across all processes
rdf = comm.reduce(rdf, op=MPI.SUM, root=0)


if rank == 0:
    # local_sum = global_sum/float(size)
    print(f"Process {rank}: Local sum = {rdf/2}")
else:
    print(f"Process {rank}: Local sum = {rdf}")