from mpi4py import MPI
from numpy import array, zeros, ones, sqrt, dot, empty
import time

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()
size = comm.Get_size()

N = 100  # Глобальный размер
local_N = N // numprocs
if rank == numprocs - 1:
    local_N += N % numprocs

# Toy-данные: диагональная A = diag(1,2,...,N), b = ones(N)
A_diag = array([i+1 + rank*local_N for i in range(local_N)], dtype='d')  # Локальная часть
b_local = ones(local_N, dtype='d')
x = zeros(local_N, dtype='d')
r = b_local.copy()
p = r.copy()

# Отложенные запросы для граничных обменов (только для не-краевых процессов)
if numprocs > 1 and 0 < rank < numprocs - 1:
    send_left = empty(1, dtype='d')
    recv_left = empty(1, dtype='d')
    send_right = empty(1, dtype='d')
    recv_right = empty(1, dtype='d')
    requests = [
        comm.Send_init([send_left, 1, MPI.DOUBLE], dest=rank-1, tag=0),
        comm.Recv_init([recv_left, 1, MPI.DOUBLE], source=rank-1, tag=0),
        comm.Send_init([send_right, 1, MPI.DOUBLE], dest=rank+1, tag=1),
        comm.Recv_init([recv_right, 1, MPI.DOUBLE], source=rank+1, tag=1)
    ]
else:
    requests = []

tol = 1e-6
max_iter = 100
rsold = dot(r, r)

start_time = time.time()
for n in range(max_iter):
    # Обмен границами асинхронно (для p и r)
    if requests:
        # Подготовка данных для отправки (последний элемент локального для left, первый для right)
        send_left[0] = p[-1] if len(p) > 0 else 0
        send_right[0] = p[0] if len(p) > 0 else 0
        MPI.Prequest.Startall(requests)
    
    # Вычисления: Ap = A * p (диагональ)
    Ap = A_diag * p
    
    # Ждём обмен
    if requests:
        MPI.Request.Waitall(requests)
        # Обновляем границы (упрощённо, для демонстрации)
        if len(p) > 0:
            p[0] += recv_left[0]  # Пример интеграции
            p[-1] += recv_right[0]
    
    # alpha = rsold / (p * Ap)
    alpha = rsold / dot(p, Ap)
    x += alpha * p
    r -= alpha * Ap
    
    rsnew = dot(r, r)
    if sqrt(rsnew) < tol:
        break
    
    beta = rsnew / rsold
    p = r + beta * p
    rsold = rsnew

time_async = time.time() - start_time

if rank == 0:
    print(f'CG converged in {n} iterations, time: {time_async:.4f}s')

# Для сравнения: блокирующая версия (упрощённо, без полного кода)
# Замени на Sendrecv_replace в аналогичном цикле и замерь.