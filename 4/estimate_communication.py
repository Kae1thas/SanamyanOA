# estimate_communication.py
import numpy as np

M, N = 10000, 500
size = 64

# Распределение строк
local_M = M // size + (1 if _ < M % size else 0 for _ in range(size))

# Передачи в matrix_vector.py
print("=== Оценка объёма коммуникаций ===")
print(f"M={M}, N={N}, size={size}")

# Scatter A
A_size = M * N * 8 / 1e6  # MB
print(f"Scatter A: {A_size:.2f} MB")

# Scatter x
x_size = M * 8 / 1e6
print(f"Scatter x: {x_size:.2f} MB")

# Reduce b_temp (N элементов)
b_size = N * 8 / 1e6
print(f"Reduce b: {b_size:.2f} MB")

# CG: Allreduce в каждой итерации
iters = N  # худший случай
allreduce_per_iter = 2 * N * 8 / 1e6  # r и q
total_cg_comm = iters * allreduce_per_iter
print(f"CG Allreduce (N итераций): {total_cg_comm:.2f} MB")

# CG Parallel: Allgatherv + Reduce_scatter
allgatherv = N * 8 / 1e6
reduce_scatter = N * 8 / 1e6
total_cg_par = iters * (allgatherv + reduce_scatter)
print(f"CG Parallel (Allgatherv + Reduce_scatter): {total_cg_par:.2f} MB")

from mpi4py import MPI
MPI.Finalize()  # <-- Критически важно!