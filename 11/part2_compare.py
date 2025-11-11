from mpi4py import MPI
from numpy import array, int32
import time

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

a = array([rank], dtype=int32)

# Версия с persistent (асинхронная)
start_p = time.time()
requests = [MPI.REQUEST_NULL for _ in range(2)]
requests[0] = comm.Send_init([a, 1, MPI.INT], dest=(rank + 1) % numprocs, tag=0)
requests[1] = comm.Recv_init([a, 1, MPI.INT], source=(rank - 1) % numprocs, tag=0)
for _ in range(10):
    MPI.Prequest.Startall(requests)
    MPI.Request.Waitall(requests)
time_p = time.time() - start_p

# Версия с Sendrecv_replace (блокирующая) — ИСПРАВЛЕНО: sendtag и recvtag
a_block = array([rank], dtype=int32)
start_b = time.time()
for _ in range(10):
    comm.Sendrecv_replace([a_block, 1, MPI.INT], 
                          dest=(rank + 1) % numprocs, 
                          source=(rank - 1) % numprocs, 
                          sendtag=0,  # Тег для отправки
                          recvtag=0)  # Тег для приёма
time_b = time.time() - start_b

# Синхронизация для вывода
comm.Barrier()

if rank == 0:
    print(f'Persistent time: {time_p:.6f}s')
    print(f'Block (Sendrecv_replace) time: {time_b:.6f}s')
    print(f'Speedup (async vs block): {time_b / time_p:.2f}x')

# Опционально: выведи финальное значение a_block для проверки
print(f'Process {rank} finished with value {a_block[0]} (should be {rank} after 10 cycles)')