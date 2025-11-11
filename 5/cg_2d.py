# cg_2d.py
from mpi4py import MPI
import numpy as np
from numpy import empty, array, int32, float64, zeros, hstack, dot
import sys
import time
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

print(f"Rank {rank}: MPI инициализировано, numprocs={numprocs}")

# Проверка: число процессов — квадрат
num_row = num_col = int32(np.sqrt(numprocs))
if num_row * num_col != numprocs:
    if rank == 0:
        print(f"Ошибка: число процессов {numprocs} не квадрат!")
    sys.exit(1)

comm_col = comm.Split(rank % num_col, rank)
comm_row = comm.Split(rank // num_col, rank)

# === N, M ===
print(f"Rank {rank}: Начинаю чтение in.dat")
if rank == 0:
    if not os.path.exists('in.dat'):
        print("Ошибка: файл in.dat не найден!")
        sys.exit(1)
    with open('in.dat', 'r') as f:
        N = array(int32(f.readline().strip()))
        M = array(int32(f.readline().strip()))
    print(f"Rank 0: Прочитал N={N}, M={M}")
else:
    N = array(0, dtype=int32)
    M = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)
comm.Bcast([M, 1, MPI.INT], root=0)
print(f"Rank {rank}: N={N}, M={M}")

# === auxiliary_arrays_determination ===
def auxiliary_arrays_determination(size, num):
    ave, res = divmod(size, num)
    rcounts = empty(num, dtype=int32)
    displs = empty(num, dtype=int32)
    for k in range(num):
        rcounts[k] = ave + 1 if k < res else ave
        displs[k] = 0 if k == 0 else displs[k-1] + rcounts[k-1]
    return rcounts, displs

if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)
else:
    rcounts_M = displs_M = None
    rcounts_N = displs_N = None

M_part = array(0, dtype=int32)
N_part = array(0, dtype=int32)

if rank in range(num_col):
    comm_row.Scatter([rcounts_N, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)
comm_col.Bcast([N_part, 1, MPI.INT], root=0)

if rank in range(0, numprocs, num_col):
    comm_col.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)
comm_row.Bcast([M_part, 1, MPI.INT], root=0)

print(f"Rank {rank}: M_part={M_part}, N_part={N_part}")

A_part = empty((M_part, N_part), dtype=float64)

# === Рассылка A ===
print(f"Rank {rank}: Начинаю загрузку матрицы A")
group = comm.Get_group()
if rank == 0:
    if not os.path.exists('AData.dat'):
        print("Ошибка: файл AData.dat не найден!")
        sys.exit(1)
    with open('AData.dat', 'r') as f:
        for m in range(num_row):
            print(f"Rank 0: Загружаю блок m={m}")
            a_temp = empty(rcounts_M[m] * N, dtype=float64)
            for j in range(rcounts_M[m]):
                for n in range(num_col):
                    for i in range(rcounts_N[n]):
                        idx = rcounts_M[m] * displs_N[n] + j * rcounts_N[n] + i
                        line = f.readline().strip()
                        if not line:
                            print("Ошибка: файл AData.dat слишком короткий!")
                            sys.exit(1)
                        a_temp[idx] = float64(line)
            print(f"Rank 0: Блок m={m} загружен, размер a_temp={a_temp.shape}")
            if m == 0:
                comm_row.Scatterv([a_temp, rcounts_M[m]*rcounts_N, rcounts_M[m]*displs_N, MPI.DOUBLE],
                                 [A_part, M_part*N_part, MPI.DOUBLE], root=0)
            else:
                group_temp = group.Range_incl([(0,0,1), (m*num_col, (m+1)*num_col-1, 1)])
                comm_temp = comm.Create(group_temp)
                rcounts_N_temp = hstack((array(0, dtype=int32), rcounts_N))
                displs_N_temp = hstack((array(0, dtype=int32), displs_N))
                comm_temp.Scatterv([a_temp, rcounts_M[m]*rcounts_N_temp, rcounts_M[m]*displs_N_temp, MPI.DOUBLE],
                                  [empty(0, dtype=float64), 0, MPI.DOUBLE], root=0)
                if rank == 0 or m*num_col <= rank < (m+1)*num_col:
                    comm_temp.Free()
                group_temp.Free()
else:
    if rank in range(num_col):
        comm_row.Scatterv([None, None, None, None], [A_part, M_part*N_part, MPI.DOUBLE], root=0)
    for m in range(1, num_row):
        group_temp = group.Range_incl([(0,0,1), (m*num_col, (m+1)*num_col-1, 1)])
        comm_temp = comm.Create(group_temp)
        if m*num_col <= rank < (m+1)*num_col:
            comm_temp.Scatterv([None, None, None, None], [A_part, M_part*N_part, MPI.DOUBLE], root=0)
            comm_temp.Free()
        group_temp.Free()

print(f"Rank {rank}: Матрица A загружена")

# === b ===
if rank == 0:
    if not os.path.exists('bData.dat'):
        print("Ошибка: файл bData.dat не найден!")
        sys.exit(1)
    b = np.loadtxt('bData.dat', dtype=float64)
else:
    b = None
b_part = empty(M_part, dtype=float64)
if rank in range(0, numprocs, num_col):
    comm_col.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], [b_part, M_part, MPI.DOUBLE], root=0)

# === x ===
if rank == 0:
    x = zeros(N, dtype=float64)
else:
    x = None
x_part = empty(N_part, dtype=float64)
if rank in range(num_col):
    comm_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], [x_part, N_part, MPI.DOUBLE], root=0)

print(f"Rank {rank}: Данные готовы, начинаю CG")

# === CG (исправленная функция) ===
def conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, N, comm_row, comm_col, rank, num_col):
    r_part = empty(N_part, dtype=float64)
    p_part = empty(N_part, dtype=float64)
    q_part = empty(N_part, dtype=float64)
    Ax_part = empty(M_part, dtype=float64)
    r_temp = empty(M_part, dtype=float64)  # Временный для r = b - Ax (M_part)

    print(f"Rank {rank}: CG: Вычисляю r = b - A x")
    # Первая итерация: r = b - A x (x = 0)
    comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)
    Ax_local = dot(A_part, x_part)  # (M_part, N_part) @ (N_part,) = (M_part,)
    comm_row.Reduce([Ax_local, M_part, MPI.DOUBLE], [Ax_part, M_part, MPI.DOUBLE], op=MPI.SUM, root=0)
    
    if rank in range(0, numprocs, num_col):
        r_temp = b_part - Ax_part  # r_temp: M_part
    comm_row.Bcast([r_temp, M_part, MPI.DOUBLE], root=0)  # Рассылка r_temp по строкам
    
    # r = A^T @ r_temp (N_part)
    r_local = dot(A_part.T, r_temp)  # (N_part, M_part) @ (M_part,) = (N_part,)
    comm_col.Reduce([r_local, N_part, MPI.DOUBLE], [r_part, N_part, MPI.DOUBLE], op=MPI.SUM, root=0)
    p_part = r_part.copy()  # p = r

    s = 1
    max_iter = min(N, 50)  # Ограничение для теста
    while s <= max_iter:
        print(f"Rank {rank}: CG итерация s={s}")
        # Ap = A @ p
        comm_col.Bcast([p_part, N_part, MPI.DOUBLE], root=0)
        Ap_local = dot(A_part, p_part)  # (M_part, N_part) @ (N_part,) = (M_part,)
        comm_row.Reduce([Ap_local, M_part, MPI.DOUBLE], [Ax_part, M_part, MPI.DOUBLE], op=MPI.SUM, root=0)

        # q = A^T @ Ap
        q_local = dot(A_part.T, Ax_part)  # (N_part, M_part) @ (M_part,) = (N_part,)
        comm_col.Reduce([q_local, N_part, MPI.DOUBLE], [q_part, N_part, MPI.DOUBLE], op=MPI.SUM, root=0)

        if rank in range(num_col):
            rsq = dot(r_part, r_part)
            if rsq < 1e-12:
                print(f"Rank {rank}: Сходимость на итерации {s}: ||r||^2 = {rsq:.2e}")
                break
            alpha = rsq / dot(p_part, q_part)
            if np.isnan(alpha) or alpha < 1e-15:
                print(f"Rank {rank}: Деление на ноль на итерации {s}, выход")
                break
            x_part += alpha * p_part
            r_part -= alpha * q_part

        s += 1

    return x_part

# === Запуск ===
start_time = time.time()
x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, N, comm_row, comm_col, rank, num_col)
end_time = time.time()

if rank == 0:
    print(f"Время выполнения: {end_time - start_time:.4f} сек")

# === Сбор x ===
if rank in range(num_col):
    comm_row.Gatherv([x_part, N_part, MPI.DOUBLE], [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank == 0:
    np.save('result_cg_2d.npy', x)
    print("Решение сохранено в result_cg_2d.npy")
    print("Первые 5 элементов:", x[:5] if N >= 5 else x)