import numpy as np
from mpi4py import MPI

def parallel_thomas(a, b, c, d, comm):
    rank = comm.Get_rank()   # Номер процесса
    size = comm.Get_size()   # Количество процессов
    n = len(b)               # Размер системы
    
    # Распределение данных
    local_n = n // size
    if rank == size - 1:
        local_n = n - rank * local_n  # Последний процесс берет остаток
    start = rank * (n // size)
    end = start + local_n
    
    # Локальные массивы
    local_a = np.zeros(local_n)
    local_b = np.zeros(local_n)
    local_c = np.zeros(local_n - 1 if rank < size - 1 else n - start - 1)
    local_d = np.zeros(local_n)
    
    # Заполнение локальных массивов
    for i in range(local_n):
        global_idx = start + i
        if global_idx > 0 and i < len(local_a):
            local_a[i] = a[global_idx - 1]
        local_b[i] = b[global_idx]
        if i < len(local_c) and global_idx < n - 1:
            local_c[i] = c[global_idx]
        local_d[i] = d[global_idx]
    
    # Прямой ход
    local_c_prime = np.zeros(local_n)
    local_d_prime = np.zeros(local_n)
    
    if local_n > 0:
        if rank == 0:
            local_c_prime[0] = local_c[0] / local_b[0] if len(local_c) > 0 else 0
            local_d_prime[0] = local_d[0] / local_b[0]
        else:
            recv_buf = np.zeros(2)
            req = comm.Irecv(recv_buf, source=rank-1, tag=0)
            req.Wait()
            prev_c_prime, prev_d_prime = recv_buf
            local_c_prime[0] = local_c[0] / (local_b[0] - local_a[0] * prev_c_prime) if len(local_c) > 0 else 0
            local_d_prime[0] = (local_d[0] - local_a[0] * prev_d_prime) / (local_b[0] - local_a[0] * prev_c_prime)
    
    for i in range(1, local_n):
        denom = local_b[i] - local_a[i] * local_c_prime[i-1]
        local_c_prime[i] = local_c[i-1] / denom if i-1 < len(local_c) else 0
        local_d_prime[i] = (local_d[i] - local_a[i] * local_d_prime[i-1]) / denom
    
    if size > 1 and rank < size - 1:
        send_buf = np.array([local_c_prime[-1], local_d_prime[-1]])
        req = comm.Isend(send_buf, dest=rank+1, tag=0)
        req.Wait()
    
    # Обратный ход
    local_x = np.zeros(local_n)
    
    if rank == size - 1:
        local_x[-1] = local_d_prime[-1]
    if size > 1 and rank < size - 1:
        recv_buf = np.zeros(1)
        req = comm.Irecv(recv_buf, source=rank+1, tag=1)
        req.Wait()
        local_x[-1] = recv_buf[0]
    
    for i in range(local_n-2, -1, -1):
        local_x[i] = local_d_prime[i] - local_c_prime[i] * local_x[i+1]
    
    if size > 1 and rank > 0:
        send_buf = np.array([local_x[0]])
        req = comm.Isend(send_buf, dest=rank-1, tag=1)
        req.Wait()
    
    # Сбор размеров локальных массивов
    sendcounts = np.array(comm.allgather(local_n))
    displacements = np.zeros(size, dtype=int)
    displacements[1:] = np.cumsum(sendcounts[:-1])
    
    # Сбор результатов
    global_x = np.zeros(n)
    comm.Allgatherv(local_x, [global_x, sendcounts, displacements, MPI.DOUBLE])
    
    # Отладочный вывод
    if rank in [0, size-1]:
        print(f"Rank {rank}: local_d_prime = {local_d_prime}")
        print(f"Rank {rank}: local_x = {local_x}")
    
    return global_x