# scripts/plot_compare_cg.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/times.csv")
cg1 = df[df['program'] == 'cg_simple']
cg2 = df[df['program'] == 'cg_parallel']

plt.figure(figsize=(10, 6))
plt.plot(cg1['procs'], cg1['time'], 'o-', label='CG Simple (Allreduce)')
plt.plot(cg2['procs'], cg2['time'], 's-', label='CG Parallel (Reduce_scatter + Allgatherv)')
plt.yscale('log')
plt.xlabel('Число процессов')
plt.ylabel('Время (с)')
plt.title('Сравнение реализаций метода сопряжённых градиентов')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/plots/cg_comparison.png", dpi=150)
plt.close()
print("Сравнение CG сохранено в results/plots/cg_comparison.png")