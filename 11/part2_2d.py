from mpi4py import MPI
from numpy import array, int32, empty, zeros

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

# 2D-массив 5x2, заполненный rank
a = array([[rank + i*numprocs + j for j in range(2)] for i in range(5)], dtype=int32)
a_recv = empty((5, 2), dtype=int32)

requests = [MPI.REQUEST_NULL for _ in range(2)]
requests[0] = comm.Send_init([a, (5, 2), MPI.INT], dest=(rank + 1) % numprocs, tag=0)
requests[1] = comm.Recv_init([a_recv, (5, 2), MPI.INT], source=(rank - 1) % numprocs, tag=0)

for _ in range(10):
    MPI.Prequest.Startall(requests)
    MPI.Request.Waitall(requests)
    a = a_recv.copy()  # Обновляем

print(f'Process {rank} finished with array:\n{a}')