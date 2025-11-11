import numpy as np
from numpy import empty, linspace, sin, pi, float64
import matplotlib.pyplot as plt
import time

# Функция для правой части f(y, t) = eps * u_xx + u * u_x + u^3
def f(y, t, h, N, u_left, u_right, eps):
    f_vec = empty(N - 1, dtype=float64)
    # Левая граница
    f_vec[0] = eps * (y[1] - 2 * y[0] + u_left(t)) / h**2 + y[0] * (y[1] - u_left(t)) / (2 * h) + y[0]**3
    # Внутренние точки
    for n in range(1, N - 2):
        f_vec[n] = eps * (y[n+1] - 2 * y[n] + y[n-1]) / h**2 + y[n] * (y[n+1] - y[n-1]) / (2 * h) + y[n]**3
    # Правая граница
    f_vec[N-2] = eps * (u_right(t) - 2 * y[N-2] + y[N-3]) / h**2 + y[N-2] * (u_right(t) - y[N-3]) / (2 * h) + y[N-2]**3
    return f_vec

# Подготовка коэффициентов трёхдиагональной матрицы для [I - alpha tau J] w = f
def diagonal_preparation(y, t, h, N, u_left, u_right, eps, tau, alpha):
    a = empty(N-1, dtype=float64)  # поддиагональ
    b = empty(N-1, dtype=float64)  # главная диагональ
    c = empty(N-1, dtype=float64)  # наддиагональ
    # Левая граница
    b[0] = 1. - alpha * tau * (-2 * eps / h**2 + (y[1] - u_left(t)) / (2 * h) + 3 * y[0]**2)
    c[0] = -alpha * tau * (eps / h**2 + y[0] / (2 * h))
    # Внутренние точки
    for n in range(1, N-2):
        a[n] = -alpha * tau * (eps / h**2 - y[n] / (2 * h))
        b[n] = 1. - alpha * tau * (-2 * eps / h**2 + (y[n+1] - y[n-1]) / (2 * h) + 3 * y[n]**2)
        c[n] = -alpha * tau * (eps / h**2 + y[n] / (2 * h))
    # Правая граница
    a[N-2] = -alpha * tau * (eps / h**2 - y[N-2] / (2 * h))
    b[N-2] = 1. - alpha * tau * (-2 * eps / h**2 + (u_right(t) - y[N-3]) / (2 * h) + 3 * y[N-2]**2)
    return a, b, c

# Метод прогонки для трёхдиагональной СЛАУ: A x = d, где A имеет a, b, c
def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    N = len(d)
    x = empty(N, dtype=float64)
    # Прямой ход
    for n in range(1, N):
        coef = a[n] / b[n-1]
        b[n] = b[n] - coef * c[n-1]
        d[n] = d[n] - coef * d[n-1]
    # Обратный ход
    x[N-1] = d[N-1] / b[N-1]
    for n in range(N-2, -1, -1):
        x[n] = (d[n] - c[n] * x[n+1]) / b[n]
    return x

# Основная функция
def solve_sequential(N=200, M=300, a=0, b=1, t0=0, T=2.0, eps=10**(-1.5), alpha=0.5):
    h = (b - a) / N
    tau = (T - t0) / M
    x = linspace(a, b, N + 1)
    t = linspace(t0, T, M + 1)
    
    # Граничные и начальные условия
    def u_left(t_val): return sin(pi * t_val)
    def u_right(t_val): return 0
    u_init = lambda x_val: sin(pi * x_val)
    
    # Инициализация u
    u = empty((M + 1, N + 1), dtype=float64)
    u[0, :] = u_init(x)
    for i in range(N + 1):
        u[0, i] = u_init(x[i])
    u[:, 0] = [u_left(ti) for ti in t]
    u[:, N] = [u_right(ti) for ti in t]
    
    start_time = time.time()
    y = u[0, 1:N].copy()  # Внутренние точки
    for m in range(M):
        # Подготовка матрицы и правой части
        a_diag, b_diag, c_diag = diagonal_preparation(y, t[m], h, N, u_left, u_right, eps, tau, alpha)
        f_right = f(y, t[m] + tau/2, h, N, u_left, u_right, eps)
        # Решение СЛАУ: w1 = inv(I - alpha tau J) * f
        w1 = consecutive_tridiagonal_matrix_algorithm(a_diag, b_diag, c_diag, f_right)
        # Обновление: y^{m+1} = y^m + tau * Re(w1)
        y = y + tau * np.real(w1)
        u[m+1, 1:N] = y
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Время выполнения последовательной версии: {exec_time:.4f} сек")
    
    # Сохранение в файл (для анализа)
    np.savetxt('u_sequential.txt', u[-1, :])  # Последний слой
    return u, exec_time, x

# Запуск и визуализация
if __name__ == "__main__":
    u, time_seq, x = solve_sequential()
    plt.figure(figsize=(10, 6))
    plt.plot(x, u[-1, :], label='u(x, T=2.0)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Решение уравнения теплопроводности (последовательная версия)')
    plt.legend()
    plt.grid(True)
    plt.savefig('solution_seq.png')
    plt.show()