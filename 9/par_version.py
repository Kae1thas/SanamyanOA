from mpi4py import MPI
import numpy as np
from numpy import empty, linspace, sin, pi, float64
import time

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()
comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()

# Метод прогонки (из seq)
def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    K = len(d)
    x = empty(K, dtype=float64)
    # Прямой ход
    for n in range(1, K):
        coef = a[n] / b[n - 1]
        b[n] -= coef * c[n - 1]
        d[n] -= coef * d[n - 1]
    # Обратный ход
    x[K - 1] = d[K - 1] / b[K - 1]
    for n in range(K - 2, -1, -1):
        x[n] = (d[n] - c[n] * x[n + 1]) / b[n]
    return x

# Локальная f(y, t + tau/2, h, local_N, left_halo_f, right_halo_f, eps)
# y: локальные внутренние точки (размер local_N)
def f_local(y, left_halo, right_halo, t, h, local_N, eps):
    f_vec = empty(local_N, dtype=float64)
    # Первая точка
    f_vec[0] = eps * (y[1] - 2 * y[0] + left_halo) / h**2 + y[0] * (y[1] - left_halo) / (2 * h) + y[0]**3
    # Внутренние
    for n in range(1, local_N - 1):
        f_vec[n] = eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h**2 + y[n] * (y[n + 1] - y[n - 1]) / (2 * h) + y[n]**3
    # Последняя
    f_vec[-1] = eps * (right_halo - 2 * y[-1] + y[-2]) / h**2 + y[-1] * (right_halo - y[-2]) / (2 * h) + y[-1]**3
    return f_vec

# Локальная подготовка a, b, c для [I - alpha tau J(y, t)]
# Размер: a, b, c по local_N (a[0]=0 не используется, c[-1]=0 не используется)
def diagonal_preparation_local(y, left_halo, right_halo, t, h, local_N, eps, tau, alpha):
    K = local_N
    a = np.zeros(K, dtype=float64)  # a[0] = 0
    b = empty(K, dtype=float64)
    c = np.zeros(K, dtype=float64)  # c[-1] = 0
    # Первая строка
    b[0] = 1. - alpha * tau * (-2 * eps / h**2 + (y[1] - left_halo) / (2 * h) + 3 * y[0]**2)
    c[0] = -alpha * tau * (eps / h**2 + y[0] / (2 * h))
    # Внутренние строки
    for k in range(1, K - 1):
        a[k] = -alpha * tau * (eps / h**2 - y[k] / (2 * h))
        b[k] = 1. - alpha * tau * (-2 * eps / h**2 + (y[k + 1] - y[k - 1]) / (2 * h) + 3 * y[k]**2)
        c[k] = -alpha * tau * (eps / h**2 + y[k] / (2 * h))
    # Последняя строка
    a[K - 1] = -alpha * tau * (eps / h**2 - y[K - 1] / (2 * h))
    b[K - 1] = 1. - alpha * tau * (-2 * eps / h**2 + (right_halo - y[K - 2]) / (2 * h) + 3 * y[K - 1]**2)
    return a, b, c

# Параллельная прогонка (пока локальная; для полной — добавить обмен coef)
def parallel_tridiagonal_matrix_algorithm(a, b, c, d):
    return consecutive_tridiagonal_matrix_algorithm(a, b, c, d)

def solve_parallel(N=200, M=300, a=0., b=1., t0=0., T=2.0, eps=10**(-1.5), alpha=0.5):
    h = (b - a) / N
    tau = (T - t0) / M
    x_global = linspace(a, b, N + 1)
    t = linspace(t0, T, M + 1)
    
    def u_left(t_val): return sin(pi * t_val)
    def u_right(t_val): return 0.
    u_init = lambda x_val: sin(pi * x_val)
    
    # Распределение внутренних точек (N-1 точек)
    if rank_cart == 0:
        inners_count = N - 1
        ave, res = divmod(inners_count, numprocs)
        rcounts = np.array([ave + 1 if k < res else ave for k in range(numprocs)], dtype='i')
        displs = np.zeros(numprocs, dtype='i')
        displs[1:] = np.cumsum(rcounts[:-1])
    else:
        rcounts = None
        displs = None
    
    N_part = comm_cart.scatter(rcounts, root=0)
    y_part = np.empty(N_part, dtype=float64)
    if rank_cart == 0:
        y_global = u_init(x_global[1:N])  # внутренние init
        comm_cart.Scatterv([y_global, rcounts, displs, MPI.DOUBLE], y_part, root=0)
    else:
        comm_cart.Scatterv([None, rcounts, displs, MPI.DOUBLE], y_part, root=0)
    
    # u_part_aux: (M+1, N_part + 2) — 0: left_halo, 1:N_part: inners, N_part+1: right_halo
    u_part_aux = np.zeros((M + 1, N_part + 2), dtype=float64)
    u_part_aux[0, 1:1 + N_part] = y_part
    # Границы на все времена (приблизительно для integer t)
    if rank_cart == 0:
        u_part_aux[:, 0] = np.array([u_left(ti) for ti in t])
    if rank_cart == numprocs - 1:
        u_part_aux[:, N_part + 1] = np.array([u_right(ti) for ti in t])
    
    # Начальный обмен halo для m=0
    def halo_exchange(m):
        if rank_cart == 0:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m, N_part:N_part + 1], 1, MPI.DOUBLE], dest=1, sendtag=0,
                recvbuf=[u_part_aux[m, N_part + 1:N_part + 2], 1, MPI.DOUBLE], source=1, recvtag=0
            )
        elif rank_cart == numprocs - 1:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m, 1:2], 1, MPI.DOUBLE], dest=numprocs - 2, sendtag=0,
                recvbuf=[u_part_aux[m, 0:1], 1, MPI.DOUBLE], source=numprocs - 2, recvtag=0
            )
        else:  # middle
            # Left exchange
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m, 1:2], 1, MPI.DOUBLE], dest=rank_cart - 1, sendtag=0,
                recvbuf=[u_part_aux[m, 0:1], 1, MPI.DOUBLE], source=rank_cart - 1, recvtag=0
            )
            # Right exchange
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m, N_part:N_part + 1], 1, MPI.DOUBLE], dest=rank_cart + 1, sendtag=0,
                recvbuf=[u_part_aux[m, N_part + 1:N_part + 2], 1, MPI.DOUBLE], source=rank_cart + 1, recvtag=0
            )
    
    halo_exchange(0)  # Для m=0
    
    start_time = time.time()
    y_part = y_part.copy()
    for m in range(M):
        # Halo для J (t[m]) и f (t[m] + tau/2)
        left_halo_j = u_part_aux[m, 0]
        right_halo_j = u_part_aux[m, N_part + 1]
        left_halo_f = u_left(t[m] + tau / 2) if rank_cart == 0 else left_halo_j
        right_halo_f = u_right(t[m] + tau / 2) if rank_cart == numprocs - 1 else right_halo_j
        
        # Подготовка
        a_diag, b_diag, c_diag = diagonal_preparation_local(
            y_part, left_halo_j, right_halo_j, t[m], h, N_part, eps, tau, alpha
        )
        f_right = f_local(y_part, left_halo_f, right_halo_f, t[m] + tau / 2, h, N_part, eps)
        
        # Решение
        w1_part = parallel_tridiagonal_matrix_algorithm(a_diag, b_diag, c_diag, f_right)
        y_part += tau * np.real(w1_part)
        
        # Обновление
        u_part_aux[m + 1, 1:1 + N_part] = y_part
        # Границы на t[m+1]
        if rank_cart == 0:
            u_part_aux[m + 1, 0] = u_left(t[m + 1])
        if rank_cart == numprocs - 1:
            u_part_aux[m + 1, N_part + 1] = u_right(t[m + 1])
        
        # Обмен halo для следующего шага (m+1)
        halo_exchange(m + 1)
    
    end_time = time.time()
    exec_time = end_time - start_time
    
    # Сбор на root (только последний слой для примера)
    inners_parts = comm_cart.gather(u_part_aux[-1, 1:1 + N_part], root=0)
    if rank_cart == 0:
        u_full = np.zeros(N + 1, dtype=float64)
        u_full[0] = u_left(T)
        u_full[-1] = u_right(T)
        u_full[1:N] = np.concatenate(inners_parts)
        np.savetxt('u_parallel.txt', u_full)
        print(f"Время параллельной версии (P={numprocs}): {exec_time:.4f} сек")
        return u_full, exec_time, x_global
    else:
        comm_cart.gather(u_part_aux[-1, 1:1 + N_part], root=0)
        return None, exec_time, None

if __name__ == "__main__":
    u, time_par, x = solve_parallel()
    if rank_cart == 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(x, u, label='u(x, T=2.0)')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'Решение (параллельная, P={numprocs})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'solution_par_{numprocs}.png')
        plt.show()

