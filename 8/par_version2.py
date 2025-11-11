from mpi4py import MPI
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from numpy import linspace, sin, pi, empty, savez, load
import matplotlib
matplotlib.use('Agg')  # Headless для WSL2, без GUI
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

# Виртуальная топология
comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()

if rank_cart == 0:
    start_time = MPI.Wtime()

# Параметры
a, b = (0, 1)
t_0, T = (0, 6)
eps = 10**(-1.5)
N, M = (200, 20_000)

x, h = linspace(a, b, N+1, retstep=True)
t, tau = linspace(t_0, T, M+1, retstep=True)

# rcounts/displs на root
if rank_cart == 0:
    ave, res = divmod(N + 1, numprocs)
    full_rcounts = np.empty(numprocs, dtype=np.int32)
    full_displs = np.empty(numprocs, dtype=np.int32)
    for k in range(numprocs):
        full_rcounts[k] = ave + (1 if k < res else 0)
        full_displs[k] = 0 if k == 0 else full_displs[k - 1] + full_rcounts[k - 1]
else:
    full_rcounts = None
    full_displs = None

# Bcast полных rcounts/displs всем (для Gatherv)
comm_cart.Bcast([full_rcounts, MPI.INT], root=0)
comm_cart.Bcast([full_displs, MPI.INT], root=0)

# Scatter N_part (локальный)
N_part = np.empty(1, dtype=np.int32)
sendbuf_part = [full_rcounts, 1, MPI.INT] if rank_cart == 0 else None
comm_cart.Scatter(sendbuf_part, [N_part, 1, MPI.INT], root=0)
N_part = N_part[0]

# rcounts_from_0/displs_from_0 для init/halo
if rank_cart == 0:
    rcounts_from_0 = full_rcounts.copy()
    displs_from_0 = np.zeros(numprocs, dtype=np.int32)
    if numprocs > 1:
        rcounts_from_0[0] += 1
        for k in range(1, numprocs - 1):
            rcounts_from_0[k] += 2
            displs_from_0[k] = full_displs[k] - 1
        rcounts_from_0[-1] += 1
        displs_from_0[-1] = full_displs[-1] - 1
else:
    rcounts_from_0 = np.empty(0, dtype=np.int32)
    displs_from_0 = np.empty(0, dtype=np.int32)

N_part_aux = np.empty(1, dtype=np.int32)
sendbuf_aux = [rcounts_from_0, 1, MPI.INT] if rank_cart == 0 else None
comm_cart.Scatter(sendbuf_aux, [N_part_aux, 1, MPI.INT], root=0)
N_part_aux = N_part_aux[0]

# displs_aux для init
displs_aux = np.empty(1, dtype=np.int32)
sendbuf_displs = [displs_from_0, 1, MPI.INT] if rank_cart == 0 else None
comm_cart.Scatter(sendbuf_displs, [displs_aux, 1, MPI.INT], root=0)
displs_aux = displs_aux[0]

# Память
u_part_aux = empty((M + 1, N_part_aux))

# Init
for n in range(N_part_aux):
    u_part_aux[0, n] = u_init(x[displs_aux + n])

if rank_cart == 0:
    for m in range(1, M + 1):
        u_part_aux[m, 0] = u_left(t[m])
elif rank_cart == numprocs - 1 and numprocs > 1:
    for m in range(1, M + 1):
        u_part_aux[m, -1] = u_right(t[m])

# Barrier перед циклом
comm_cart.Barrier()
print(f"Rank {rank_cart}: Init done, start loop")

# Цикл
for m in range(M):
    # Вычисления
    for n in range(1, N_part_aux - 1):
        d2 = (u_part_aux[m, n + 1] - 2 * u_part_aux[m, n] + u_part_aux[m, n - 1]) / h ** 2
        d1 = (u_part_aux[m, n + 1] - u_part_aux[m, n - 1]) / (2 * h)
        u_part_aux[m + 1, n] = u_part_aux[m, n] + eps * tau * d2 + tau * u_part_aux[m, n] * d1 + tau * (u_part_aux[m, n]) ** 3
    
    # Обмен границами
    if numprocs > 1:
        if rank_cart == 0:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m + 1, -2], 1, MPI.DOUBLE],
                dest=1, sendtag=0,
                recvbuf=[u_part_aux[m + 1, -1:], 1, MPI.DOUBLE],
                source=1, recvtag=MPI.ANY_TAG, status=None
            )
        elif rank_cart == numprocs - 1:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m + 1, 1], 1, MPI.DOUBLE],
                dest=numprocs - 2, sendtag=0,
                recvbuf=[u_part_aux[m + 1, 0:1], 1, MPI.DOUBLE],
                source=numprocs - 2, recvtag=MPI.ANY_TAG, status=None
            )
        else:
            # Левый
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m + 1, 1], 1, MPI.DOUBLE],
                dest=rank_cart - 1, sendtag=0,
                recvbuf=[u_part_aux[m + 1, 0:1], 1, MPI.DOUBLE],
                source=rank_cart - 1, recvtag=MPI.ANY_TAG, status=None
            )
            # Правый
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m + 1, -2], 1, MPI.DOUBLE],
                dest=rank_cart + 1, sendtag=0,
                recvbuf=[u_part_aux[m + 1, -1:], 1, MPI.DOUBLE],
                source=rank_cart + 1, recvtag=MPI.ANY_TAG, status=None
            )
    
    if m % 5000 == 0:
        print(f"Rank {rank_cart}: After Sendrecv m={m}")

# Barrier после цикла
comm_cart.Barrier()
print(f"Rank {rank_cart}: Loop done, start Gatherv")

# Gatherv: один вызов на всех
if numprocs == 1:
    u_T = u_part_aux[M].copy()
else:
    # Подготовка send_slice (без halo, len=N_part)
    if rank_cart == 0:
        send_slice = u_part_aux[M, 0:N_part]
    elif rank_cart == numprocs - 1:
        send_slice = u_part_aux[M, 1:N_part + 1]  # Исключаем left halo
    else:
        send_slice = u_part_aux[M, 1:N_part + 1]  # Исключаем оба halo
    send_count = N_part
    
    recvbuf = [empty(N + 1), full_rcounts, full_displs, MPI.DOUBLE] if rank_cart == 0 else None
    comm_cart.Gatherv([send_slice, send_count, MPI.DOUBLE], recvbuf, root=0)
    if rank_cart == 0:
        u_T = recvbuf[0]

comm_cart.Barrier()
print(f"Rank {rank_cart}: Gatherv done")

if rank_cart == 0:
    end_time = MPI.Wtime()
    print(f"N={N}, M={M}")
    print(f"Number of MPI processes: {numprocs}")
    print(f"Elapsed time: {end_time - start_time:.4f} sec.")
    print(f"Final u mean: {u_T.mean():.4f} (should be ~0, no NaN)")
    savez("results_par2.npz", x=x, u_final=u_T, t_final=t[-1])

# Визуализация (headless, только save)
if rank_cart == 0:
    results = load("results_par2.npz")
    x_vis = results["x"]
    u_final = results["u_final"]
    if np.isnan(u_final).any():
        print("Внимание: NaN в результатах!")
    plt.figure(figsize=(10, 6))
    plt.plot(x_vis, u_final, 'r-', lw=2)
    plt.ylim(-2, 2)
    plt.title(f"Финальное решение (Sendrecv, t={results['t_final']:.2f})")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True)
    plt.savefig("final_par2.png", dpi=150)
    plt.close()  # Закрываем без show
    print("График сохранён в final_par2.png")

# Финал
comm_cart.Free()
comm.Barrier()
if rank_cart == 0:
    print("Все завершено!")