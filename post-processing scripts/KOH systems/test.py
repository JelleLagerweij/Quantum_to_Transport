<<<<<<< HEAD
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
=======
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD

# Get the predefined error handler MPI.ERRORS_ARE_FATAL
errhandler = MPI.ERRORS_ARE_FATAL

# Set the error handler
comm.Set_errhandler(errhandler)
i= comm.rank

print(f'entre loop, i = {i}', flush=True)

if i == 4:
    print(f'on rank {i}, things should error')
    try:
        # Simulate an error (replace this with your actual code)
        raise ValueError("Oehoe")

    except Exception as e:
        # This exception will trigger the error handler since we've set ERRORS_ARE_FATAL
        pass

print('done', flush=True)
>>>>>>> c2363854429ab53ed94759b525d9264bcd387798
