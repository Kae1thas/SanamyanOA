from mpi4py import MPI
import numpy as np
import time
import os  # Для OMP

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def generate_local_matrix(N_global, local_rows):
    """Локальная часть симметричной положительно определённой матрицы A"""
    A_local = np.zeros((local_rows, N_global), dtype=np.float64)
    for i in range(local_rows):
        global_i = rank * local_rows + i
        if global_i >= N_global:
            global_i = N_global - 1  # Безопасно для остатка
        for j in range(N_global):
            A_local[i, j] = 1.0 / (1.0 + abs(global_i - j))
        A_local[i, global_i] += 2.0 * N_global  # Диагональное доминирование
    return A_local

def conjugate_gradient_hybrid(A_local, b_local, x0_global, N_global, tol=1e-5, max_iter=50):
    """Гибридный CG: Полные x/p на всех, локальные r/Ap, OpenMP в dot"""
    x = x0_global.copy() if x0_global is not None else np.zeros(N_global, dtype=np.float64)
    local_rows = len(b_local)
    local_x_start = rank * local_rows
    local_x_end = min(local_x_start + local_rows, N_global)
    
    # Инициализация residual
    r_local = b_local - np.dot(A_local, x)  # Полный x
    rsold_local = np.dot(r_local, r_local)
    rsold = np.array([rsold_local], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, rsold, op=MPI.SUM)
    rsold_global = rsold[0]
    
    if np.sqrt(rsold_global) < tol:
        if rank == 0:
            print("Уже сходится на старте!")
        return x
    
    # p = r: собираем глобальный r
    r_global = np.empty(N_global, dtype=np.float64)
    comm.Allgather(r_local, r_global)
    p = r_global.copy()
    
    local_time_start = time.time() if rank == 0 else None
    converged_iter = 0
    
    for i in range(max_iter):
        # Ap_local = A_local @ p (OpenMP ускоряет!)
        Ap_local = np.dot(A_local, p)
        
        # Глобальный (p, Ap): sum over local p_slice * Ap_local
        p_local = p[local_x_start:local_x_end]
        pAp_local = np.dot(p_local, Ap_local)
        pAp = np.array([pAp_local], dtype=np.float64)
        pAp_global = np.zeros(1, dtype=np.float64)
        comm.Allreduce(pAp, pAp_global, op=MPI.SUM)
        alpha = rsold_global / pAp_global[0]
        
        # Update x и r_local
        x += alpha * p
        r_local -= alpha * Ap_local
        rsnew_local = np.dot(r_local, r_local)
        rsnew = np.array([rsnew_local], dtype=np.float64)
        rsnew_global = np.zeros(1, dtype=np.float64)
        comm.Allreduce(rsnew, rsnew_global, op=MPI.SUM)
        
        if np.sqrt(rsnew_global[0]) < tol:
            converged_iter = i + 1
            break
        
        beta = rsnew_global[0] / rsold_global
        p = r_global + beta * p  # Временно, обновим r_global ниже
        rsold_global = rsnew_global[0]
        
        # Обновляем r_global для следующей итерации
        comm.Allgather(r_local, r_global)
        p = r_global + beta * p  # Правильный p
    
    if rank == 0:
        total_time = time.time() - local_time_start
        print(f"CG завершён: {converged_iter if converged_iter > 0 else max_iter} итераций, время: {total_time:.2f} с")
        return x, total_time
    else:
        return None, None

# Основной запуск
if __name__ == "__main__":
    N = 4000  # Глобальный размер (для замеров)
    local_rows = N // size
    if rank == size - 1:
        local_rows += N % size
    
    A_part = generate_local_matrix(N, local_rows)
    b_part = np.ones(local_rows, dtype=np.float64) * (2 * N)  # b для x ≈ ones
    x0 = np.zeros(N, dtype=np.float64)
    
    threads = os.environ.get('OMP_NUM_THREADS', '1')
    if rank == 0:
        print(f"Гибридный CG: {size} процессов, {threads} потоков на процесс")
    
    result, exec_time = conjugate_gradient_hybrid(A_part, b_part, x0, N)
    
    # Проверка нормы на root (локальная часть)
    if rank == 0 and result is not None:
        local_start = 0
        local_end = min(N // size, N)
        Ax_local = np.dot(A_part, result)
        error_local = np.linalg.norm(Ax_local - b_part)
        print(f"Локальная норма ошибки на root: {error_local:.2e}")