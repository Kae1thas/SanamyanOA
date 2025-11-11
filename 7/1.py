import numpy as np
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from parallel import parallel_thomas

def sequential_thomas(a, b, c, d):
    n = len(b)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    # Прямой ход
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    # Обратный ход
    x = np.zeros(n)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

def plot_performance(ns, speedup, efficiency, p):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ns, speedup, marker='o')
    plt.xscale('log')
    plt.xlabel('n')
    plt.ylabel('Ускорение')
    plt.title(f'Ускорение (p={p})')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(ns, efficiency, marker='o')
    plt.xscale('log')
    plt.xlabel('n')
    plt.ylabel('Эффективность')
    plt.title(f'Эффективность (p={p})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'speedup_p{p}.png')
    plt.close()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ns = [100, 1000, 10000, 100000, 1000000]
    speedup = []
    efficiency = []
    
    for n in ns:
        # Инициализация тестовых данных
        a = np.full(n-1, -1.0)
        b = np.full(n, 2.0)
        c = np.full(n-1, -1.0)
        d = np.zeros(n)
        exact_x = np.ones(n)  # Точное решение
        
        # Последовательный алгоритм
        start_time = time.time()
        x_seq = sequential_thomas(a, b, c, d)
        seq_time = time.time() - start_time
        
        # Ошибка последовательного алгоритма
        seq_error = np.max(np.abs(x_seq - exact_x))
        if rank == 0:
            print(f"Ошибка последовательного алгоритма для n={n}: {seq_error}")
        
        # Параллельный алгоритм
        comm.Barrier()
        start_time = time.time()
        x_par = parallel_thomas(a, b, c, d, comm)
        par_time = time.time() - start_time
        
        # Ошибка параллельного алгоритма
        par_error = np.max(np.abs(x_par - exact_x))
        if rank == 0:
            print(f"Ошибка параллельного алгоритма для n={n}, p={size}: {par_error}")
            print(f"n={n}: global_x[:10] = {x_par[:10]}")
            print(f"n={n}: global_x[-10:] = {x_par[-10:]}")
        
        # Расчет ускорения и эффективности
        if rank == 0:
            speedup_n = seq_time / par_time if par_time > 0 else 1.0
            efficiency_n = speedup_n / size
            speedup.append(speedup_n)
            efficiency.append(efficiency_n)
    
    # Вывод ускорения и эффективности
    if rank == 0:
        print("\nУскорение и эффективность:")
        print(f"Процессов: {size}")
        for i, n in enumerate(ns):
            print(f"  n={n}: Ускорение={speedup[i]:.2f}, Эффективность={efficiency[i]:.2f}")
        
        # Построение графиков
        plot_performance(ns, speedup, efficiency, size)

if __name__ == "__main__":
    main()