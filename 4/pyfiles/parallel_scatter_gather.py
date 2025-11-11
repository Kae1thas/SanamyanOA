# parallel_scatter_gather.py
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

    # Проверка делимости
    if M % size != 0:
        if rank == 0:
            print("ERROR: M must be divisible by number of processes")
        MPI.Finalize()
        return

    local_M = M // size

    # Загрузка A и x
    if rank == 0:
        A = np.loadtxt('AData.dat').reshape(M, N)
        x = np.loadtxt('xData.dat')
    else:
        A = None
        x = None

    # Распределение A
    A_part = np.empty((local_M, N), dtype=np.float64)
    sendcounts = [local_M * N] * size
    displs = [i * local_M * N for i in range(size)]
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
    sendcounts_b = [local_M] * size
    displs_b = [i * local_M for i in range(size)]
    comm.Gatherv(b_part, [b, sendcounts_b, displs_b, MPI.DOUBLE], root=0)

    if rank == 0:
        np.savetxt('Results_parallel2.dat', b, fmt='%.6f')
        elapsed = MPI.Wtime() - start_time
        print(f"TIME_ELAPSED: {elapsed:.6f}")

    MPI.Finalize()

if __name__ == "__main__":
    main()