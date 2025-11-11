import numpy as np
import time
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    N = 10000  # Размер для заметного времени
    A = np.random.rand(N, N)
    x = np.random.rand(N)
    
    threads_list = [1, 4, 8, 16]  # Тестируем разные (адаптируй под твои ядра: nproc в htop)
    results = []
    
    for t in threads_list:
        os.environ['OMP_NUM_THREADS'] = str(t)
        os.environ['OPENBLAS_NUM_THREADS'] = str(t)  # Для OpenBLAS
        os.environ['MKL_NUM_THREADS'] = str(t)       # Если MKL (редко в Ubuntu)
        
        start = time.time()
        y = np.dot(A, x)  # Тест операции
        end = time.time()
        duration = end - start
        results.append((t, duration))
        print(f"Потоки: {t}, Время: {duration:.3f} с")
    
    print("Результаты:")
    for t, d in results:
        print(f"  {t} потоков: {d:.3f} с (speedup: {results[0][1]/d:.2f}x)")
    
    np.show_config()  # Конфиг один раз в конце