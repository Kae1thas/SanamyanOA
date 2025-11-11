import numpy as np
import time
import os  # Добавил для проверки env
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    N = 10000
    A = np.random.rand(N, N)
    x = np.random.rand(N)
    start = time.time()
    y = np.dot(A, x)  # Здесь OpenMP через BLAS
    end = time.time()
    print(f"Время на rank 0: {end - start:.2f} с")
    # Исправленная проверка потоков
    threads = os.environ.get('OPENBLAS_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', 'по умолчанию (все ядра)'))
    print(f"Потоки (из env): {threads}")
    # Альтернатива: np.show_config() печатает в консоль, но не возвращает dict
    print("Конфиг NumPy:")
    np.show_config()