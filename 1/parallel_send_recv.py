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

assert M % size == 0, "M must be divisible by size"
local_M = M // size
print(f"Rank {rank}: local_M calculated: {local_M}")
sys.stdout.flush()

if rank == 0:
    print(f"Rank {rank}: Loading A and x")
    sys.stdout.flush()
    A = np.loadtxt('AData.dat').reshape(M, N)
    x = np.loadtxt('xData.dat')
    for i in range(1, size):
        start_row = i * local_M
        A_part = A[start_row:start_row + local_M]
        print(f"Rank {rank}: Sending A_part to rank {i}")
        sys.stdout.flush()
        comm.Send([A_part, MPI.DOUBLE], dest=i, tag=i)
    A_part = A[0:local_M]
    print(f"Rank {rank}: Set own A_part")
    sys.stdout.flush()
else:
    print(f"Rank {rank}: Preparing to receive A_part")
    sys.stdout.flush()
    A_part = np.empty((local_M, N), dtype=np.float64)
    comm.Recv([A_part, MPI.DOUBLE], source=0, tag=rank)
    print(f"Rank {rank}: Received A_part")
    sys.stdout.flush()
    x = None  

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

if rank == 0:
    print(f"Rank {rank}: Collecting results")
    sys.stdout.flush()
    b = np.empty(M, dtype=np.float64)
    b[0:local_M] = b_part
    for i in range(1, size):
        start_row = i * local_M
        b_part_recv = np.empty(local_M, dtype=np.float64)
        print(f"Rank {rank}: Receiving b_part from rank {i}")
        sys.stdout.flush()
        comm.Recv([b_part_recv, MPI.DOUBLE], source=i, tag=i)
        b[start_row:start_row + local_M] = b_part_recv
    np.savetxt('Results_parallel.dat', b, fmt='%.6f')
    print(f"Rank {rank}: Results saved. Parallel time: {time.time() - start_time} seconds")
    sys.stdout.flush()
else:
    print(f"Rank {rank}: Sending b_part to root")
    sys.stdout.flush()
    comm.Send([b_part, MPI.DOUBLE], dest=0, tag=rank)
    print(f"Rank {rank}: Sent b_part")
    sys.stdout.flush()

print(f"Rank {rank}: Finished")
sys.stdout.flush()