# parallel_scatter_gather_variable.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    # Чтение in.dat
    if rank == 0:
        with open('in.dat', 'r') as f:
            line = f.readline().strip()
            M, N = map(int, line.split())
        start_time = MPI.Wtime()
    else:
        M = None
        N = None
        start_time = None

    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)

    # Динамическое распределение
    if rank == 0:
        base = M // size
        remainder = M % size
        rcounts = [base + (1 if i < remainder else 0) for i in range(size)]
        displs = [0]
        for i in range(1, size):
            displs.append(displs[-1] + rcounts[i-1])
        local_M = rcounts[0]
    else:
        rcounts = None
        displs = None
        local_M = None

    local_M = comm.bcast(local_M, root=0)
    rcounts = comm.bcast(rcounts, root=0)
    displs = comm.bcast(displs, root=0)

    # Загрузка A и x
    if rank == 0:
        A = np.loadtxt('AData.dat').reshape(M, N)
        x = np.loadtxt('xData.dat')
    else:
        A = None
        x = None

    # Распределение A
    A_part = np.empty((local_M, N), dtype=np.float64)
    sendcounts = [r * N for r in rcounts]
    comm.Scatterv([A, sendcounts, displs, MPI.DOUBLE], A_part, root=0)

    # Рассылка x
    x = comm.bcast(x, root=0)

    # Умножение
    b_part = np.dot(A_part, x)

    # Сбор результата
    if rank == 0:
        b = np.empty(M, dtype=np.float64)
    else:
        b = None
    comm.Gatherv(b_part, [b, rcounts, displs, MPI.DOUBLE], root=0)

    if rank == 0:
        np.savetxt('Results_parallel_variable.dat', b, fmt='%.6f')
        elapsed = MPI.Wtime() - start_time
        print(f"TIME_ELAPSED: {elapsed:.6f}")

    MPI.Finalize()

if __name__ == "__main__":
    main()