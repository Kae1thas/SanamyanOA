from numpy import empty, linspace, sin, pi
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import savez, load

def u_init(x):
    return sin(3 * pi * (x - 1/6))

def u_left(t):
    return -1

def u_right(t):
    return +1

# Замер времени
start_time = time.time()

# Параметры задачи
a, b = (0, 1)
t_0, T = (0, 6)
eps = 10**(-1.5)
N, M = (200, 20_000)  # Маленькие для теста; для больших — 800, 300000

x, h = linspace(a, b, N+1, retstep=True)
t, tau = linspace(t_0, T, M+1, retstep=True)

# Массив сеточных значений
u = empty((M+1, N+1))

# Начальные условия
for n in range(N+1):
    u[0, n] = u_init(x[n])

# Граничные условия
for m in range(M+1):
    u[m, 0] = u_left(t[m])
    u[m, N] = u_right(t[m])

# Основной цикл: явная схема
for m in range(M):
    for n in range(1, N):
        d2 = (u[m, n+1] - 2*u[m, n] + u[m, n-1]) / h**2
        d1 = (u[m, n+1] - u[m, n-1]) / (2*h)
        u[m+1, n] = u[m, n] + eps*tau*d2 + tau*u[m, n]*d1 + tau*(u[m, n])**3

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.4f} sec.")

# Сохранение результатов
savez("results_seq.npz", x=x, u=u)

# Визуализация: анимация
results = load("results_seq.npz")
x = results["x"]
u = results["u"]
M_total = u.shape[0]

fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    ax.set_ylim(-2, 2)
    line, = ax.plot(x, u[i * M_total // 100, :], color='blue', lw=1)
    return line

anim = FuncAnimation(fig, animate, interval=40, frames=100)
anim.save("results_seq.gif", dpi=300)
plt.show()
print("Анимация сохранена в results_seq.gif")