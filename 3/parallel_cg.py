from mpi4py import MPI
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def auxiliary_arrays_determination(M, numprocs):
    ave, res = divmod(M, numprocs-1 if numprocs>1 else 1)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)
    rcounts[0] = 0
    displs[0] = 0
    for k in range(1, numprocs):
        if k <= res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k-1] + int(rcounts[k-1])
    return rcounts, displs

def read_input():
    if rank == 0:
        with open('in.dat','r') as f:
            N = np.array(int(f.readline().strip()), dtype=np.int32)
            M = np.array(int(f.readline().strip()), dtype=np.int32)
    else:
        N = np.array(0, dtype=np.int32)
        M = np.array(0, dtype=np.int32)
    comm.Bcast([N, MPI.INT], root=0)
    comm.Bcast([M, MPI.INT], root=0)
    return int(N), int(M)

def conjugate_gradient_method(A_part, b_part, x_part, N, N_part, rcounts_N, displs_N):
    N = int(N)
    x = np.empty(N, dtype=np.float64)
    p = np.empty(N, dtype=np.float64)
    r_part = np.empty(N_part, dtype=np.float64)
    q_part = np.empty(N_part, dtype=np.float64)
    ScalP = np.array(0.0, dtype=np.float64)
    ScalP_temp = np.empty(1, dtype=np.float64)

    s = 1
    p_part = np.zeros(N_part, dtype=np.float64)

    while s <= N:
        if s == 1:
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE],
                            [x, rcounts_N, displs_N, MPI.DOUBLE])
            r_temp = A_part.T.dot(A_part.dot(x) - b_part)
            comm.Reduce_scatter([r_temp, MPI.DOUBLE],
                                [r_part, MPI.DOUBLE],
                                recvcounts=rcounts_N, op=MPI.SUM)
        else:
            r_part -= q_part / ScalP

        ScalP_temp[0] = np.dot(r_part, r_part)
        comm.Allreduce([ScalP_temp, MPI.DOUBLE], [ScalP, MPI.DOUBLE], op=MPI.SUM)
        p_part += r_part / ScalP

        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p, rcounts_N, displs_N, MPI.DOUBLE])

        q_temp = A_part.T.dot(A_part.dot(p))
        comm.Reduce_scatter([q_temp, MPI.DOUBLE],
                            [q_part, MPI.DOUBLE],
                            recvcounts=rcounts_N, op=MPI.SUM)

        ScalP_temp[0] = np.dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, MPI.DOUBLE], [ScalP, MPI.DOUBLE], op=MPI.SUM)

        x_part -= p_part / ScalP

        s += 1

    return x_part

def main():
    N, M = read_input()

    if rank == 0:
        rcounts_M, displs_M = auxiliary_arrays_determination(M, size)
        rcounts_N, displs_N = auxiliary_arrays_determination(N, size)
    else:
        rcounts_M, displs_M = None, None
        rcounts_N = np.empty(size, dtype=np.int32)
        displs_N = np.empty(size, dtype=np.int32)

    comm.Bcast([rcounts_N, MPI.INT], root=0)
    comm.Bcast([displs_N, MPI.INT], root=0)

    M_part = np.array(0, dtype=np.int32)
    if rank == 0:
        tmp = np.empty(1, dtype=np.int32)
        tmp[0] = rcounts_M[rank]
        M_part = np.array(rcounts_M[rank], dtype=np.int32)
    comm.Scatter([rcounts_M if rank == 0 else None, MPI.INT], [M_part, MPI.INT], root=0)

    if rank == 0:
        with open('AData.dat','r') as f:
            for k in range(1, size):
                rows = rcounts_M[k]
                if rows == 0:
                    continue
                A_part = np.empty((rows, N), dtype=np.float64)
                for j in range(rows):
                    for i in range(N):
                        A_part[j,i] = float(f.readline().strip())
                comm.Send([A_part, rows*N, MPI.DOUBLE], dest=k, tag=77)
        A_part = np.empty((M_part, N), dtype=np.float64)
    else:
        A_part = np.empty((int(M_part), N), dtype=np.float64)
        if int(M_part) > 0:
            comm.Recv([A_part, int(M_part)*N, MPI.DOUBLE], source=0, tag=77)

    if rank == 0:
        b = np.empty(M, dtype=np.float64)
        with open('bData.dat','r') as f:
            for j in range(M):
                b[j] = float(f.readline().strip())
    else:
        b = None
    b_part = np.empty(int(M_part), dtype=np.float64)
    comm.Scatterv([b, rcounts_M, MPI.DOUBLE], [b_part, int(M_part), MPI.DOUBLE], root=0)

    if rank == 0:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = None
    x_part = np.empty(int(rcounts_N[rank]), dtype=np.float64)
    comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], [x_part, int(rcounts_N[rank]), MPI.DOUBLE], root=0)

    t0 = MPI.Wtime()
    x_part = conjugate_gradient_method(A_part, b_part, x_part, N, int(rcounts_N[rank]), rcounts_N, displs_N)
    t1 = MPI.Wtime()

    if rank == 0:
        x_res = np.empty(N, dtype=np.float64)
    else:
        x_res = None
    comm.Gatherv([x_part, int(rcounts_N[rank]), MPI.DOUBLE], [x_res, rcounts_N, displs_N, MPI.DOUBLE], root=0)

    if rank == 0:
        print("Time (core):", t1-t0, "s")
        A_full = np.empty((M,N), dtype=np.float64)
        with open('AData.dat','r') as f:
            for j in range(M):
                for i in range(N):
                    A_full[j,i] = float(f.readline().strip())
        b_full = np.empty(M, dtype=np.float64)
        with open('bData.dat','r') as f:
            for j in range(M):
                b_full[j] = float(f.readline().strip())
        x_np, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
        err = np.linalg.norm(x_res - x_np) / (np.linalg.norm(x_np)+1e-16)
        print("rel err to numpy lstsq:", err)

if __name__ == "__main__":
    main()
