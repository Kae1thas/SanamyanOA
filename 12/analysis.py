import matplotlib.pyplot as plt
import numpy as np

# Твои данные (замени на реальные из замеров)
nodes = [1, 4, 8, 16]
times_hybrid = [0.1, 0.2, 0.1, 0.4]  # Время гибрид (с)
times_mpi = [0.023, 0.025, 0.027, 0.027]    # Время MPI (с)
T1_h = times_hybrid[0]
T1_m = times_mpi[0]
speedup_h = [T1_h / t for t in times_hybrid]  # Ускорение гибрид
speedup_m = [T1_m / t for t in times_mpi]     # Ускорение MPI
efficiency_h = [s / n for s, n in zip(speedup_h, nodes)]  # Эффективность гибрид

# График 1: Время выполнения
plt.figure(figsize=(8, 6))
plt.plot(nodes, times_hybrid, 'b-o', linewidth=2, markersize=6, label='Гибрид')
plt.plot(nodes, times_mpi, 'r-o', linewidth=2, markersize=6, label='MPI')
plt.xlabel('Число узлов (n)')
plt.ylabel('Время (с)')
plt.title('Время выполнения метода сопряжённых градиентов')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(nodes)
plt.tight_layout()
plt.savefig('time_lab12.png', dpi=300, bbox_inches='tight')
plt.show()

# График 2: Ускорение
plt.figure(figsize=(8, 6))
plt.plot(nodes, speedup_h, 'b-o', linewidth=2, markersize=6, label='Гибрид')
plt.plot(nodes, speedup_m, 'r-o', linewidth=2, markersize=6, label='MPI')
plt.plot(nodes, nodes, 'k--', linewidth=1, alpha=0.7, label='Идеальное (S=n)')
plt.xlabel('Число узлов (n)')
plt.ylabel('Ускорение (Speedup)')
plt.title('Ускорение гибридной реализации')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(nodes)
plt.tight_layout()
plt.savefig('speedup_lab12.png', dpi=300, bbox_inches='tight')
plt.show()

# График 3: Эффективность
plt.figure(figsize=(8, 6))
plt.plot(nodes, efficiency_h, 'g-o', linewidth=2, markersize=6, label='Гибрид')
plt.plot(nodes, [1.0] * len(nodes), 'k--', linewidth=1, alpha=0.7, label='Идеал (E=1)')
plt.xlabel('Число узлов (n)')
plt.ylabel('Эффективность (Efficiency)')
plt.title('Эффективность гибридной реализации')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(nodes)
plt.tight_layout()
plt.savefig('efficiency_lab12.png', dpi=300, bbox_inches='tight')
plt.show()