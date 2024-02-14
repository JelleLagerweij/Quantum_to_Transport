from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
n_array = 3

sendbuf = np.arange(2*n_array).reshape((2, n_array))*rank
if rank == 0:
    print(sendbuf, type(sendbuf))
else:
    print(sendbuf)
sendbuf = comm.gather(sendbuf, root=0)

if rank == 0:
    sendbuf = np.concatenate(sendbuf, axis=1)
    print(sendbuf, type(sendbuf))
else:
    print(sendbuf)
# if rank == 0:
#     sendbuf = np.arange(size*n_array)
#     print(sendbuf, type(sendbuf), sendbuf.shape)
#     sendbuf = np.array_split(sendbuf, size, axis=0)
#     print(sendbuf, type(sendbuf))
# else:
#     sendbuf = None
# recvbuf = comm.scatter(sendbuf, root=0)
# print(recvbuf, 'rank', rank)

