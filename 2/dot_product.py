import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M = 10 

if rank == 0:
    start_time_seq = time.time()
    a = np.arange(1, M + 1, dtype=np.float64)
    seq_dot = np.dot(a, a)
    seq_time = time.time() - start_time_seq
    print(f"Последовательное скалярное произведение: {seq_dot}")
    print(f"Время последовательной версии: {seq_time:.6f} секунд")
else:
    a = None

local_M = M // size
rcounts = [local_M] * size
remainder = M % size
for i in range(remainder):
    rcounts[i] += 1
displs = [0] * size
for i in range(1, size):
    displs[i] = displs[i-1] + rcounts[i-1]

a_part = np.empty(rcounts[rank], dtype=np.float64)

start_time_par = time.time()

comm.Scatterv([a, rcounts, displs, MPI.DOUBLE], a_part, root=0)

local_dot = np.dot(a_part, a_part)

global_dot_reduce = np.array(0.0, dtype=np.float64) if rank == 0 else None
comm.Reduce(local_dot, global_dot_reduce, op=MPI.SUM, root=0)
if rank == 0:
    reduce_time = time.time() - start_time_par
    print(f"Параллельное (Reduce): {global_dot_reduce}")
    print(f"Время параллельной версии (Reduce): {reduce_time:.6f} секунд")

start_time_allreduce = time.time()
global_dot_allreduce = np.array(0.0, dtype=np.float64)
comm.Allreduce(local_dot, global_dot_allreduce, op=MPI.SUM)
allreduce_time = time.time() - start_time_allreduce
print(f"Процесс {rank}: Параллельное (Allreduce): {global_dot_allreduce}")
print(f"Процесс {rank}: Время параллельной версии (Allreduce): {allreduce_time:.6f} секунд")