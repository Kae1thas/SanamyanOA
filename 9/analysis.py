# plot_results.py
# Построение графиков по готовым данным из таблицы
# Сохраняет: images/execution_time.png, images/speedup.png, images/efficiency.png

import numpy as np
import matplotlib.pyplot as plt
import os

# === ДАННЫЕ ИЗ ТАБЛИЦЫ (вставь свои реальные замеры) ===
procs = [1, 2, 4, 8, 16]
times = [0.4831, 0.2553, 0.1402, 0.1219, 0.2501]  # в секундах

# === Расчёт ускорения и эффективности ===
t_serial = times[0]
speedup = [t_serial / t for t in times]
efficiency = [s / p for s, p in zip(speedup, procs)]

# === Создание папки ===
os.makedirs("images", exist_ok=True)

# === 1. График времени выполнения ===
plt.figure(figsize=(6, 4.5))
plt.plot(procs, times, 'o-', color='red', linewidth=2, markersize=8, label='Время выполнения')
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Время (секунды)', fontsize=12)
plt.title('Зависимость времени выполнения от числа процессов', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("images/execution_time.png", dpi=200, bbox_inches='tight')
plt.close()

# === 2. График ускорения ===
plt.figure(figsize=(6, 4.5))
plt.plot(procs, speedup, 'o-', color='blue', linewidth=2, markersize=8, label='Ускорение')
plt.plot(procs, procs, '--', color='gray', linewidth=1.5, label='Линейное ускорение')
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Ускорение', fontsize=12)
plt.title('Ускорение (Speedup)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("images/speedup.png", dpi=200, bbox_inches='tight')
plt.close()

# === 3. График эффективности ===
plt.figure(figsize=(6, 4.5))
plt.plot(procs, efficiency, 's-', color='green', linewidth=2, markersize=8, label='Эффективность')
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Эффективность', fontsize=12)
plt.title('Эффективность параллелизма', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("images/efficiency.png", dpi=200, bbox_inches='tight')
plt.close()

# === 4. Общий график (все три) ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(procs, times, 'o-', color='red', linewidth=2, markersize=6)
plt.xlabel('Процессов')
plt.ylabel('Время (с)')
plt.title('Время выполнения')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(procs, speedup, 'o-', color='blue', linewidth=2, markersize=6, label='Ускорение')
plt.plot(procs, procs, '--', color='gray', linewidth=1.5, label='Линейное')
plt.xlabel('Процессов')
plt.ylabel('Ускорение')
plt.title('Ускорение')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(procs, efficiency, 's-', color='green', linewidth=2, markersize=6)
plt.xlabel('Процессов')
plt.ylabel('Эффективность')
plt.title('Эффективность')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("images/performance_analysis.png", dpi=200, bbox_inches='tight')
plt.close()

# === Вывод ===
print("Графики успешно построены и сохранены в папку images/:")
print("   • execution_time.png")
print("   • speedup.png")
print("   • efficiency.png")
print("   • performance_analysis.png (все вместе)")
print("\nГотово! Вставь графики в отчёт.")