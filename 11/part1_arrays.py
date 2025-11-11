from mpi4py import MPI
from numpy import array, int32, zeros

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

# Массив из 10 элементов: [rank, rank+1, ..., rank+9]
a = array([rank + i for i in range(10)], dtype=int32)
b = zeros(10, dtype=int32)

requests = [MPI.REQUEST_NULL for _ in range(2)]

if rank == 0:
    requests[0] = comm.Isend([a, 10, MPI.INT], dest=numprocs-1, tag=0)
    requests[1] = comm.Irecv([b, 10, MPI.INT], source=1, tag=0)
elif rank == numprocs - 1:
    requests[0] = comm.Isend([a, 10, MPI.INT], dest=numprocs-2, tag=0)
    requests[1] = comm.Irecv([b, 10, MPI.INT], source=0, tag=0)
else:
    requests[0] = comm.Isend([a, 10, MPI.INT], dest=rank-1, tag=0)
    requests[1] = comm.Irecv([b, 10, MPI.INT], source=rank+1, tag=0)

MPI.Request.Waitall(requests)
print(f'Process {rank} got array: {b}')