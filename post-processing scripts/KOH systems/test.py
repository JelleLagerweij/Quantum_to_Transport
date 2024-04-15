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