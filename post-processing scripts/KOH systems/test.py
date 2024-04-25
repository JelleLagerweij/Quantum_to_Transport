import numpy as np
from mpi4py import MPI
from mpi4py.util.pkl5 import _dumps, _loads

# Initialize MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Define a large array only on rank 0
if rank == 0:
    large_array = np.full(1024**3, rank, dtype='i4')  # Create a large array
else:
    large_array = None

# Serialize the large array on rank 0
serialized_data = None
if rank == 0:
    serialized_data = _dumps(large_array)

# Scatter the serialized data from rank 0 to all other processes
received_serialized_data = comm.scatter(serialized_data, root=0)

# Deserialize the received data back into an array
received_array = _loads(received_serialized_data)

print("Rank:", rank, "Received array size:", received_array.size)