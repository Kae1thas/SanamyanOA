# lab5_matvec_2d.py
# Параллельная реализация умножения матрицы на вектор с двумерной декомпозицией
# Запуск: mpirun -np 16 python lab5_matvec_2d.py
# Требуется: файлы in.dat, AData.dat, xData.dat в текущей директории

from mpi4py import MPI
import numpy as np
from numpy import empty, array, int32, float64, hstack, sqrt
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_size()

# Этап 1.1: Проверка, что numprocs — квадрат числа
num_row = num_col = int32(sqrt(numprocs))
if num_row * num_col != numprocs:
    if rank == 0:
        print(f"Ошибка: число процессов {numprocs} не является квадратом натурального числа!")
    sys.exit(1)

# Этап 1.2: Создание коммуникаторов
comm_col = comm.Split(rank % num_col, rank)
comm_row = comm.Split(rank // num_col, rank)

# Этап 1.3: Чтение размеров и распределение M_part, N_part
if rank == 0:
    f = open('in.dat', 'r')
    N = array(int32(f.readline().strip()))
    M = array(int32(f.readline().strip()))
    f.close()
else:
    N = array(0, dtype=int32)
    M = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)
comm.Bcast([M, 1, MPI.INT], root=0)

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

# Рассылка N_part по первой строке, затем Bcast по столбцам
if rank in range(num_col):
    comm_row.Scatter([rcounts_N, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)
comm_col.Bcast([N_part, 1, MPI.INT], root=0)

# Рассылка M_part по первому столбцу, затем Bcast по строкам
if rank in range(0, numprocs, num_col):
    comm_col.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)
comm_row.Bcast([M_part, 1, MPI.INT], root=0)

# Выделяем память под блок матрицы
A_part = empty((M_part, N_part), dtype=float64)

# Этап 1.4: Распределение матрицы A
group = comm.Get_group()

if rank == 0:
    f = open('AData.dat', 'r')
    for m in range(num_row):
        a_temp = empty(rcounts_M[m] * N, dtype=float64)
        for j in range(rcounts_M[m]):
            for n in range(num_col):
                for i in range(rcounts_N[n]):
                    idx = rcounts_M[m] * displs_N[n] + j * rcounts_N[n] + i
                    a_temp[idx] = float64(f.readline().strip())

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
    f.close()
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

# Распределение вектора x
if rank == 0:
    x = empty(N, dtype=float64)
    f = open('xData.dat', 'r')
    for i in range(N):
        x[i] = float64(f.readline().strip())
    f.close()
else:
    x = None

x_part = empty(N_part, dtype=float64)

if rank in range(num_col):
    comm_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE], [x_part, N_part, MPI.DOUBLE], root=0)
comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)

# Этап 1.5: Локальное умножение
b_part_temp = np.dot(A_part, x_part)
b_part = empty(M_part, dtype=float64)
comm_row.Reduce([b_part_temp, M_part, MPI.DOUBLE], [b_part, M_part, MPI.DOUBLE], op=MPI.SUM, root=0)

# Сбор результата на процессах 0, num_col, 2*num_col, ...
if rank == 0:
    b = empty(M, dtype=float64)
else:
    b = None

if rank in range(0, numprocs, num_col):
    comm_col.Gatherv([b_part, M_part, MPI.DOUBLE], [b, rcounts_M, displs_M, MPI.DOUBLE], root=0)

if rank == 0:
    print("Умножение завершено. Первые 5 элементов результата:", b[:5] if M >= 5 else b)
    np.save('result_matvec.npy', b)
