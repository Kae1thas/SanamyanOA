from mpi4py import MPI
import numpy as np
import time
import sys 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank}: Starting initialization")
sys.stdout.flush()

if rank == 0:
    start_time = time.time()
    print(f"Rank {rank}: Reading in.dat")
    sys.stdout.flush()
    with open('in.dat', 'r') as f:
        M, N = map(int, f.read().split())
else:
    M = None
    N = None

print(f"Rank {rank}: Before bcast M and N")
sys.stdout.flush()
M = comm.bcast(M, root=0)
N = comm.bcast(N, root=0)
print(f"Rank {rank}: After bcast, M={M}, N={N}")
sys.stdout.flush()

# Динамическое распределение строк
if rank == 0:
    rcounts = []
    displs = [0]
    base = M // size
    remainder = M % size
    for i in range(size):
        extra = 1 if i < remainder else 0
        rcounts.append(base + extra)
        if i < size - 1:
            displs.append(displs[-1] + rcounts[-1])
    local_M = rcounts[0]
else:
    rcounts = None
    displs = None
    local_M = None

local_M = comm.bcast(local_M, root=0)
print(f"Rank {rank}: local_M calculated: {local_M}")
sys.stdout.flush()

if rank == 0:
    print(f"Rank {rank}: Loading A and x")
    sys.stdout.flush()
    A = np.loadtxt('AData.dat').reshape(M, N)
    x = np.loadtxt('xData.dat')
else:
    A = None
    x = None

print(f"Rank {rank}: Preparing for Scatterv A_part")
sys.stdout.flush()
A_part = np.empty((local_M, N), dtype=np.float64)

# Распределение rcounts и displs
rcounts = comm.bcast(rcounts, root=0)
displs = comm.bcast(displs, root=0)
sendcounts = [r * N for r in rcounts]

if rank == 0:
    comm.Scatterv([A, sendcounts, displs, MPI.DOUBLE], A_part, root=0)
else:
    comm.Scatterv(None, A_part, root=0)

print(f"Rank {rank}: Received A_part via Scatterv")
sys.stdout.flush()

print(f"Rank {rank}: Before bcast x")
sys.stdout.flush()
x = comm.bcast(x, root=0)
print(f"Rank {rank}: After bcast x")
sys.stdout.flush()

print(f"Rank {rank}: Computing b_part = A_part * x")
sys.stdout.flush()
b_part = np.dot(A_part, x)
print(f"Rank {rank}: b_part computed")
sys.stdout.flush()

print(f"Rank {rank}: Preparing for Gatherv b_part")
sys.stdout.flush()

if rank == 0:
    b = np.empty(M, dtype=np.float64)
    comm.Gatherv(b_part, [b, rcounts, displs, MPI.DOUBLE], root=0)
    np.savetxt('Results_parallel_variable.dat', b, fmt='%.6f')
    print(f"Rank {rank}: Results saved. Parallel time: {time.time() - start_time} seconds")
    sys.stdout.flush()
else:
    comm.Gatherv(b_part, None, root=0)

print(f"Rank {rank}: Finished")
sys.stdout.flush()