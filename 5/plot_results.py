import numpy as np
import matplotlib.pyplot as plt

procs = [1, 4, 9, 16]
times_1d = [0.2925, 0.1392, 0.1105, 0.05] 
times_2d = [0.0004, 0.0066, 0.0096, 0.0106] 

T1 = times_2d[0]
speedup_2d = [T1 / t for t in times_2d]
speedup_1d = [T1 / t for t in times_1d]

efficiency_2d = [s / p for s, p in zip(speedup_2d, procs)]
efficiency_1d = [s / p for s, p in zip(speedup_1d, procs)]

# 1. График эффективности
plt.figure(figsize=(8, 6))
plt.plot(procs, efficiency_1d, 'o-', label='1D декомп')
plt.plot(procs, efficiency_2d, 's-', label='2D декомп')
plt.title('Эффективность')
plt.xlabel('Процессов')
plt.ylabel('E(p)')
plt.legend()
plt.grid(True)
plt.savefig('efficiency.png')
plt.close()

# 2. График времени выполнения
plt.figure(figsize=(8, 6))
plt.plot(procs, times_1d, 'o-', label='1D декомп')
plt.plot(procs, times_2d, 's-', label='2D декомп')
plt.title('Время выполнения')
plt.xlabel('Процессов')
plt.ylabel('Время (с)')
plt.legend()
plt.grid(True)
plt.savefig('execution_time.png')
plt.close()

# 3. График ускорения
plt.figure(figsize=(8, 6))
plt.plot(procs, speedup_1d, 'o-', label='1D декомп')
plt.plot(procs, speedup_2d, 's-', label='2D декомп')
plt.title('Ускорение')
plt.xlabel('Процессов')
plt.ylabel('S(p)')
plt.legend()
plt.grid(True)
plt.savefig('speedup.png')
plt.close()