from mpi4py import MPI
from numpy import array, int32
import time  # Для простых замеров (опционально)

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

a = array([rank], dtype=int32)  # Теперь 1D с одним элементом
b = array([0], dtype=int32)     # Аналогично

requests = [MPI.REQUEST_NULL for _ in range(2)]

if rank == 0:
    requests[0] = comm.Isend([a, 1, MPI.INT], dest=numprocs-1, tag=0)
    requests[1] = comm.Irecv([b, 1, MPI.INT], source=1, tag=0)
elif rank == numprocs - 1:
    requests[0] = comm.Isend([a, 1, MPI.INT], dest=numprocs-2, tag=0)
    requests[1] = comm.Irecv([b, 1, MPI.INT], source=0, tag=0)
else:
    requests[0] = comm.Isend([a, 1, MPI.INT], dest=rank-1, tag=0)
    requests[1] = comm.Irecv([b, 1, MPI.INT], source=rank+1, tag=0)

MPI.Request.Waitall(requests)
print(f'Process {rank} got number {b[0]}')