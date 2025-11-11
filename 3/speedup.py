import matplotlib.pyplot as plt

# упрощённая версия
times_simple = [
    0.2925,           # 2 процесса
    0.1392,           # 4 процесса
    0.1105            # 8 процессов
]

# полная версия
times_full = [
    0.5006,           # 2 процесса
    0.2433,           # 4 процесса
    0.1553            # 8 процессов
]

processes = [2, 4, 8]

speedup_simple = [times_simple[0] / t for t in times_simple]
speedup_full   = [times_full[0] / t   for t in times_full]

eff_simple = [s/p for s, p in zip(speedup_simple, processes)]
eff_full   = [s/p for s, p in zip(speedup_full, processes)]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(processes, speedup_simple, marker='o', label='Simplified')
plt.plot(processes, speedup_full, marker='o', label='Full')
plt.xlabel('Number of processes')
plt.ylabel('Speedup')
plt.title('Speedup vs Processes')
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(processes, eff_simple, marker='o', label='Simplified')
plt.plot(processes, eff_full, marker='o', label='Full')
plt.xlabel('Number of processes')
plt.ylabel('Efficiency')
plt.title('Efficiency vs Processes')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("cg_speedup_efficiency.png")
