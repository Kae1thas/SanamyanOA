import matplotlib.pyplot as plt

processes = [1, 4, 9, 16]
times_cart = [0.0024, 0.2224, 0.2078, 1.1293] # ЛР6
times_base_full = [0.005, 0.5006, 0.2433, 0.1553] # ЛР3 полная
times_base_simple = [0.002, 0.2925, 0.1392, 0.1105] # ЛР3 упрощённая

# Ускорение (относительно ЛР3 полной для 1 процесса: 0.0290)
speedup_base_full = [0.0290 / t for t in times_base_full]
speedup_base_simple = [0.0290 / t for t in times_base_simple]
speedup_cart = [0.0290 / t for t in times_cart]

# Эффективность
efficiency_base_full = [s / p for s, p in zip(speedup_base_full, processes)]
efficiency_base_simple = [s / p for s, p in zip(speedup_base_simple, processes)]
efficiency_cart = [s / p for s, p in zip(speedup_cart, processes)]

# 1. График эффективности
plt.figure(figsize=(8, 6))
plt.plot(processes, efficiency_base_full, 'o-', label="Базовый CG полный (ЛР3)")
plt.plot(processes, efficiency_base_simple, 's-', label="Базовый CG упрощённый (ЛР3)")
plt.plot(processes, efficiency_cart, '^-', label="Топология (ЛР6)")
plt.title('Эффективность')
plt.xlabel('Число процессов')
plt.ylabel('E(p)')
plt.legend()
plt.grid(True)
plt.savefig('efficiency.png')
plt.close()

# 2. График времени выполнения
plt.figure(figsize=(8, 6))
plt.plot(processes, times_base_full, 'o-', label="Базовый CG полный (ЛР3)")
plt.plot(processes, times_base_simple, 's-', label="Базовый CG упрощённый (ЛР3)")
plt.plot(processes, times_cart, '^-', label="Топология (ЛР6)")
plt.title('Время выполнения')
plt.xlabel('Число процессов')
plt.ylabel('Время (с)')
plt.legend()
plt.grid(True)
plt.savefig('execution_time.png')
plt.close()

# 3. График ускорения
plt.figure(figsize=(8, 6))
plt.plot(processes, speedup_base_full, 'o-', label="Базовый CG полный (ЛР3)")
plt.plot(processes, speedup_base_simple, 's-', label="Базовый CG упрощённый (ЛР3)")
plt.plot(processes, speedup_cart, '^-', label="Топология (ЛР6)")
plt.title('Ускорение')
plt.xlabel('Число процессов')
plt.ylabel('S(p)')
plt.legend()
plt.grid(True)
plt.savefig('speedup.png')
plt.close()