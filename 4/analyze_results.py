# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# Папки
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Читаем CSV
df = pd.read_csv(os.path.join(RESULTS_DIR, "times.csv"))

# Базовое время
t_seq = df[df['procs'] == 1]['time'].iloc[0]

# Убираем ERROR
df = df[df['time'] != 'ERROR']
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['speedup'] = pd.to_numeric(df['speedup'], errors='coerce')
df['efficiency'] = pd.to_numeric(df['efficiency'], errors='coerce')
df = df.dropna(subset=['time'])

# -----------------------------
# 1. График времени выполнения
# -----------------------------
plt.figure(figsize=(8, 6))
for prog in df['program'].unique():
    if prog == 'sequential': 
        continue
    sub = df[df['program'] == prog]
    plt.plot(sub['procs'], sub['time'], 'o-', label=prog)
plt.yscale('log')
plt.xlabel('Число процессов')
plt.ylabel('Время (с, лог. шкала)')
plt.title('Время выполнения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "execution_time.png"), dpi=150)
plt.close()

# -----------------------------
# 2. График ускорения
# -----------------------------
plt.figure(figsize=(8, 6))
for prog in df['program'].unique():
    if prog == 'sequential': 
        continue
    sub = df[df['program'] == prog]
    plt.plot(sub['procs'], sub['speedup'], 's-', label=prog)
plt.plot([1, 64], [1, 64], 'k--', label='Идеальное')
plt.xlabel('Число процессов')
plt.ylabel('Ускорение')
plt.title('Ускорение')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "speedup.png"), dpi=150)
plt.close()

# -----------------------------
# 3. График эффективности
# -----------------------------
plt.figure(figsize=(8, 6))
for prog in df['program'].unique():
    if prog == 'sequential': 
        continue
    sub = df[df['program'] == prog]
    plt.plot(sub['procs'], sub['efficiency'], '^-', label=prog)
plt.axhline(1.0, color='k', linestyle='--')
plt.xlabel('Число процессов')
plt.ylabel('Эффективность')
plt.title('Эффективность')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "efficiency.png"), dpi=150)
plt.close()

print("Графики сохранены:")
print("  → results/plots/execution_time.png")
print("  → results/plots/speedup.png")
print("  → results/plots/efficiency.png")