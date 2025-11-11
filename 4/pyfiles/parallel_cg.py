# parallel_cg.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def auxiliary_arrays_determination(M, numprocs):
    if numprocs <= 1:
        return [M], [0]
    ave, res = divmod(M, numprocs - 1)
    rcounts = [0] * numprocs
    displs = [0] * numprocs
    for k in range(1, numprocs):
        rcounts[k] = ave + (1 if k <= res else 0)
        displs[k] = displs[k-1] + rcounts[k-1]
    return rcounts, displs

def read_input():
    if rank == 0:
        with open('in.dat', 'r') as f:
            line = f.readline().strip()
            M, N = map(int, line.split())  # M=10000, N=500
    else:
        M = None
        N = None
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    return N, M  # N — размер x, M — размер b

def conjugate_gradient_method(A_part, b_part, x_part, N, N_part, rcounts_N, displs_N):
    x = np.empty(N, dtype=np.float64)
    p = np.empty(N, dtype=np.float64)
    r_part = np.empty(N_part, dtype=np.float64)
    q_part = np.empty(N_part, dtype=np.float64)
    ScalP = np.array(0.0, dtype=np.float64)
    ScalP_temp = np.empty(1, dtype=np.float64)
    p_part = np.zeros(N_part, dtype=np.float64)

    for s in range(1, N + 1):
        if s == 1:
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE], [x, rcounts_N, displs_N, MPI.DOUBLE])
            r_temp = A_part.T.dot(A_part.dot(x) - b_part)
            comm.Reduce_scatter([r_temp, MPI.DOUBLE], [r_part, MPI.DOUBLE], recvcounts=rcounts_N, op=MPI.SUM)
        else:
            r_part -= q_part / ScalP

        ScalP_temp[0] = np.dot(r_part, r_part)
        comm.Allreduce([ScalP_temp, MPI.DOUBLE], [ScalP, MPI.DOUBLE], op=MPI.SUM)
        p_part += r_part / ScalP

        comm.Allgatherv([p_part, N_part, MPI.DOUBLE], [p, rcounts_N, displs_N, MPI.DOUBLE])

        q_temp = A_part.T.dot(A_part.dot(p))
        comm.Reduce_scatter([q_temp, MPI.DOUBLE], [q_part, MPI.DOUBLE], recvcounts=rcounts_N, op=MPI.SUM)

        ScalP_temp[0] = np.dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, MPI.DOUBLE], [ScalP, MPI.DOUBLE], op=MPI.SUM)

        x_part -= p_part / ScalP

    return x_part

def main():
    N, M = read_input()  # N=500, M=10000

    # Определяем распределение — ВСЕМ процессам
    if rank == 0:
        rcounts_M, displs_M = auxiliary_arrays_determination(M, size)
        rcounts_N, displs_N = auxiliary_arrays_determination(N, size)
    else:
        rcounts_M = [0] * size
        displs_M = [0] * size
        rcounts_N = np.empty(size, dtype=np.int32)
        displs_N = np.empty(size, dtype=np.int32)

    # Рассылаем rcounts_M и displs_M всем
    rcounts_M = comm.bcast(rcounts_M, root=0)
    displs_M = comm.bcast(displs_M, root=0)
    rcounts_N = comm.bcast(rcounts_N, root=0)
    displs_N = comm.bcast(displs_N, root=0)

    # Локальные размеры
    M_part = rcounts_M[rank]
    N_part = rcounts_N[rank]

    # Распределяем A
    if rank == 0:
        A_full = np.loadtxt('AData.dat').reshape(M, N)
        for k in range(1, size):
            rows = rcounts_M[k]
            if rows > 0:
                start = displs_M[k]
                A_part = A_full[start:start+rows]
                comm.Send([A_part, rows * N, MPI.DOUBLE], dest=k, tag=77)
        A_part = A_full[0:rcounts_M[0]]
    else:
        rows = rcounts_M[rank]
        A_part = np.empty((rows, N), dtype=np.float64)
        if rows > 0:
            comm.Recv([A_part, rows * N, MPI.DOUBLE], source=0, tag=77)

    # Распределяем b
    if rank == 0:
        b = np.loadtxt('bData.dat')
    else:
        b = None
    b_part = np.empty(M_part, dtype=np.float64)
    comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], [b_part, M_part, MPI.DOUBLE], root=0)

    # Распределяем x
    if rank == 0:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = None
    x_part = np.zeros(N_part, dtype=np.float64)
    comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], [x_part, N_part, MPI.DOUBLE], root=0)

    t0 = MPI.Wtime()
    x_part = conjugate_gradient_method(A_part, b_part, x_part, N, N_part, rcounts_N, displs_N)
    t1 = MPI.Wtime()

    # Собираем x
    x_res = np.empty(N, dtype=np.float64) if rank == 0 else None
    comm.Gatherv([x_part, N_part, MPI.DOUBLE], [x_res, rcounts_N, displs_N, MPI.DOUBLE], root=0)

    if rank == 0:
        print(f"TIME_ELAPSED: {t1 - t0:.6f}")

    MPI.Finalize()

if __name__ == "__main__":
    main()