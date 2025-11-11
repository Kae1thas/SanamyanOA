import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    start_time_seq = time.time()
    with open('in.dat', 'r') as f:
        M = int(f.readline().strip())
        N = int(f.readline().strip())
    
    A = np.loadtxt('AData.dat', dtype=np.float64).reshape(M, N)
    x = np.loadtxt('xData.dat', dtype=np.float64)
    
    b_seq = np.dot(A.T, x)
    seq_time = time.time() - start_time_seq
    print(f"Последовательный результат: {b_seq}")
    print(f"Время последовательной версии: {seq_time:.6f} секунд")
else:
    M = None
    N = None
    A = None
    x = None

start_time_par = time.time()

M = comm.bcast(M, root=0)
N = comm.bcast(N, root=0)

local_M = M // size
rcounts = [local_M * N] * size  
remainder = M % size
for i in range(remainder):
    rcounts[i] += N
displs = [0] * size
for i in range(1, size):
    displs[i] = displs[i-1] + rcounts[i-1]

rcounts_x = [local_M] * size
for i in range(remainder):
    rcounts_x[i] += 1
displs_x = [0] * size
for i in range(1, size):
    displs_x[i] = displs_x[i-1] + rcounts_x[i-1]

A_part = np.empty(rcounts[rank] // N * N, dtype=np.float64).reshape(rcounts_x[rank], N)
x_part = np.empty(rcounts_x[rank], dtype=np.float64)

if rank == 0:
    A_flat = A.flatten()
else:
    A_flat = None
comm.Scatterv([A_flat, rcounts, displs, MPI.DOUBLE], A_part, root=0)
comm.Scatterv([x, rcounts_x, displs_x, MPI.DOUBLE], x_part, root=0)

b_temp = np.dot(A_part.T, x_part)

b = np.empty(N, dtype=np.float64) if rank == 0 else None
comm.Reduce([b_temp, N, MPI.DOUBLE], [b, N, MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:
    par_time = time.time() - start_time_par
    print(f"Параллельный результат: {b}")
    print(f"Время параллельной версии: {par_time:.6f} секунд")