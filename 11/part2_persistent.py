from mpi4py import MPI
from numpy import array, int32, empty

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

a = array([rank], dtype=int32)  # Отправляем rank
a_recv = empty(1, dtype=int32)

requests = [MPI.REQUEST_NULL for _ in range(2)]
requests[0] = comm.Send_init([a, 1, MPI.INT], dest=(rank + 1) % numprocs, tag=0)
requests[1] = comm.Recv_init([a_recv, 1, MPI.INT], source=(rank - 1) % numprocs, tag=0)

for _ in range(10):
    MPI.Prequest.Startall(requests)
    MPI.Request.Waitall(requests)
    a[0] = a_recv[0]  # Обновляем для следующей итерации

print(f'Process {rank} finished with value {a[0]}')