from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def auxiliary_arrays_determination(M, numprocs):
    if numprocs <= 0:
        numprocs = 1
    ave, res = divmod(M, numprocs)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)
    for k in range(numprocs):
        rcounts[k] = ave + (1 if k < res else 0)
        displs[k] = (displs[k-1] + rcounts[k-1]) if k > 0 else 0
    return rcounts, displs

def read_input():
    if rank == 0:
        with open('in.dat', 'r') as f:
            line = f.readline().strip()
            M, N = map(int, line.split())  # M строк, N столбцов
    else:
        M = None
        N = None
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    return N, M  # N — размер x, M — размер b

def conjugate_gradient_method(A_part, b_part, x, N):
    r = np.empty(N, dtype=np.float64)
    q = np.empty(N, dtype=np.float64)
    p = np.zeros(N, dtype=np.float64)
    s = 1
    while s <= N:
        if s == 1:
            r_temp = A_part.T.dot(A_part.dot(x) - b_part)
            comm.Allreduce([r_temp, MPI.DOUBLE], [r, MPI.DOUBLE], op=MPI.SUM)
        else:
            r -= q / np.dot(p, q)
        p += r / np.dot(r, r)
        q_temp = A_part.T.dot(A_part.dot(p))
        comm.Allreduce([q_temp, MPI.DOUBLE], [q, MPI.DOUBLE], op=MPI.SUM)
        x -= p / np.dot(p, q)
        s += 1
    return x

def main():
    N, M = read_input()  # N=500, M=10000
    rcounts_M, displs_M = auxiliary_arrays_determination(M, size)

    # Распределение A
    if rank == 0:
        with open('AData.dat', 'r') as f:
            for k in range(size):
                rows = rcounts_M[k]
                if rows == 0:
                    if k != 0:
                        comm.Send([np.empty(0), MPI.DOUBLE], dest=k, tag=11)
                    continue
                A_part = np.empty((rows, N), dtype=np.float64)
                for j in range(rows):
                    line = f.readline().strip()
                    A_part[j] = np.fromstring(line, sep=' ', dtype=np.float64)
                if k == 0:
                    local_A = A_part
                else:
                    comm.Send([A_part, rows * N, MPI.DOUBLE], dest=k, tag=11)
    else:
        rows = rcounts_M[rank]
        local_A = np.empty((rows, N), dtype=np.float64)
        if rows > 0:
            comm.Recv([local_A, rows * N, MPI.DOUBLE], source=0, tag=11)

    # Распределение b
    if rank == 0:
        b = np.loadtxt('bData.dat', dtype=np.float64)
    else:
        b = None
    b_part = np.empty(rcounts_M[rank], dtype=np.float64)
    comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], [b_part, rcounts_M[rank], MPI.DOUBLE], root=0)

    # Решение
    x = np.zeros(N, dtype=np.float64)

    t0 = MPI.Wtime()
    x = conjugate_gradient_method(local_A, b_part, x, N)
    t1 = MPI.Wtime()

    if rank == 0:
        print(f"TIME_ELAPSED: {t1 - t0:.6f}")

    MPI.Finalize()

if __name__ == "__main__":
    main()