import numpy as np
import matplotlib.pyplot as plt

# Примерные данные (ЗАМЕНИ НА СВОИ!)
procs = [1, 2, 4, 8]  # Число процессов

# 1. Данные для CG: время выполнения (async vs sync)
time_cg_async = [1.0, 0.6, 0.3, 0.2]  # Время async CG (с)
time_cg_sync = [1.0, 0.8, 0.5, 0.4]   # Время sync CG (с)

# Ускорение CG (относительно 1 процесса)
speedup_cg_async = np.array(time_cg_async[0]) / np.array(time_cg_async)
speedup_cg_sync = np.array(time_cg_sync[0]) / np.array(time_cg_sync)

# 2. Данные для кольцевого обмена: время vs. итерации (10,50,100,500)
iters = [10, 50, 100, 500]
time_ring_persistent = [0.00015, 0.0006, 0.0012, 0.005]  # Persistent (async)
time_ring_block = [0.00002, 0.0001, 0.0002, 0.001]       # Block (Sendrecv_replace)

# График 1: Ускорение CG — отдельный PNG
plt.figure(figsize=(8, 6))
plt.plot(procs, speedup_cg_sync, 'b-o', label='Синхронная CG')
plt.plot(procs, speedup_cg_async, 'r-o', label='Асинхронная CG')
plt.xlabel('Число процессов')
plt.ylabel('Ускорение')
plt.title('Ускорение метода сопряжённых градиентов')
plt.legend()
plt.grid(True)
plt.savefig('cg_speedup.png', dpi=300)
plt.close()  # Закрываем, чтобы не мешать следующему

# График 2: Время выполнения CG — отдельный PNG
plt.figure(figsize=(8, 6))
plt.plot(procs, time_cg_sync, 'b-o', label='Синхронная CG')
plt.plot(procs, time_cg_async, 'r-o', label='Асинхронная CG')
plt.xlabel('Число процессов')
plt.ylabel('Время (с)')
plt.title('Время выполнения CG')
plt.legend()
plt.grid(True)
plt.savefig('cg_time.png', dpi=300)
plt.close()

# График 3: Сравнение кольцевого обмена — отдельный PNG
plt.figure(figsize=(8, 6))
plt.plot(iters, time_ring_block, 'g-o', label='Блокирующая (Sendrecv_replace)')
plt.plot(iters, time_ring_persistent, 'm-o', label='Persistent (async)')
plt.xlabel('Число итераций')
plt.ylabel('Время (с)')
plt.title('Сравнение кольцевого обмена')
plt.legend()
plt.grid(True)
plt.xscale('log')  # Логарифмическая шкала для итераций
plt.savefig('ring_comparison.png', dpi=300)
plt.close()

print("Готово! Сохранены: cg_speedup.png, cg_time.png, ring_comparison.png")