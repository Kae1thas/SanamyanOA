from mpi4py import MPI
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Подавляем RuntimeWarnings
from numpy import linspace, sin, pi, hstack, empty, savez, load
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def u_init(x):
    return sin(3 * pi * (x - 1/6))

def u_left(t):
    return -1

def u_right(t):
    return +1

if rank == 0:
    start_time = MPI.Wtime()

# Параметры: стабильные из seq-примера (N=200, M=20k; для лекции большой M — нестабильно!)
a, b = (0, 1)
t_0, T = (0, 6)
eps = 10**(-1.5)
N, M = (200, 20_000)  # Стабильно; для теста большой M=300_000 — blow-up!

x, h = linspace(a, b, N+1, retstep=True)
t, tau = linspace(t_0, T, M+1, retstep=True)

# rcounts и displs как np.array (int32)
if rank == 0:
    ave, res = divmod(N + 1, numprocs)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)
    for k in range(numprocs):
        rcounts[k] = ave + (1 if k < res else 0)
        displs[k] = 0 if k == 0 else displs[k - 1] + rcounts[k - 1]
else:
    rcounts = np.empty(0, dtype=np.int32)
    displs = np.empty(0, dtype=np.int32)

# Scatter для N_part (count=1!)
N_part = np.empty(1, dtype=np.int32)
sendbuf_part = [rcounts, 1, MPI.INT] if rank == 0 else None
comm.Scatter(sendbuf_part, [N_part, 1, MPI.INT], root=0)
N_part = N_part[0]

# rcounts_from_0 и displs_from_0
if rank == 0:
    rcounts_from_0 = rcounts.copy()
    displs_from_0 = np.zeros(numprocs, dtype=np.int32)
    if numprocs > 1:
        rcounts_from_0[0] += 1
        for k in range(1, numprocs - 1):
            rcounts_from_0[k] += 2
            displs_from_0[k] = displs[k] - 1
        rcounts_from_0[-1] += 1
        displs_from_0[-1] = displs[-1] - 1
else:
    rcounts_from_0 = np.empty(0, dtype=np.int32)
    displs_from_0 = np.empty(0, dtype=np.int32)

# Scatter для N_part_aux (count=1!)
N_part_aux = np.empty(1, dtype=np.int32)
sendbuf_aux = [rcounts_from_0, 1, MPI.INT] if rank == 0 else None
comm.Scatter(sendbuf_aux, [N_part_aux, 1, MPI.INT], root=0)
N_part_aux = N_part_aux[0]

# Память
if rank == 0:
    u = empty((M + 1, N + 1))
    for n in range(N + 1):
        u[0, n] = u_init(x[n])
    for m in range(M + 1):
        u[m, 0] = u_left(t[m])
        u[m, N] = u_right(t[m])
else:
    u = empty((0, 0))

u_part = empty(N_part)
u_part_aux = empty(N_part_aux)

# Цикл
for m in range(M):
    # Scatterv
    sendbuf_v = [u[m], rcounts_from_0, displs_from_0, MPI.DOUBLE] if rank == 0 else None
    comm.Scatterv(sendbuf_v, [u_part_aux, N_part_aux, MPI.DOUBLE], root=0)
    
    # Вычисления (warnings подавлены)
    for n in range(1, N_part_aux - 1):
        d2 = (u_part_aux[n + 1] - 2 * u_part_aux[n] + u_part_aux[n - 1]) / h ** 2
        d1 = (u_part_aux[n + 1] - u_part_aux[n - 1]) / (2 * h)
        u_part[n - 1] = u_part_aux[n] + eps * tau * d2 + tau * u_part_aux[n] * d1 + tau * (u_part_aux[n]) ** 3
    
    # hstack границ для краёв
    if rank == 0:
        internals = u_part[:N_part - 1] if N_part > 1 else np.empty(0)
        u_part = hstack((u_left(t[m + 1]), internals))
    elif rank == numprocs - 1 and numprocs > 1:
        internals = u_part[:N_part - 1] if N_part > 1 else np.empty(0)
        u_part = hstack((internals, u_right(t[m + 1])))
    
    # Gatherv
    recvbuf_v = [u[m + 1], rcounts, displs, MPI.DOUBLE] if rank == 0 else None
    comm.Gatherv([u_part, N_part, MPI.DOUBLE], recvbuf_v, root=0)

if rank == 0:
    end_time = MPI.Wtime()
    print(f"N={N}, M={M}")
    print(f"Number of MPI processes: {numprocs}")
    print(f"Elapsed time: {end_time - start_time:.4f} sec.")
    print(f"Final u mean: {u[-1].mean():.4f} (should be ~0, no NaN)")
    savez("results_par1.npz", x=x, u_final=u[-1], t_final=t[-1])

# Визуализация на root
if rank == 0:
    results = load("results_par1.npz")
    x_vis = results["x"]
    u_final = results["u_final"]
    if np.isnan(u_final).any():
        print("Внимание: NaN в результатах! Уменьшите M или увеличьте h (N меньше).")
    plt.figure(figsize=(10, 6))
    plt.plot(x_vis, u_final, 'b-', lw=2)
    plt.ylim(-2, 2)
    plt.title(f"Финальное решение (t={results['t_final']:.2f})")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True)
    plt.savefig("final_par1.png", dpi=150)
    plt.show()
    print("График сохранён в final_par1.png")