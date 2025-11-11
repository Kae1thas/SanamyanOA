import numpy as np
import matplotlib.pyplot as plt
import os

# Функция для чтения времён (с fallback на примерные данные, включая p=16)
def load_times(filename, default_data=None):
    if os.path.exists(filename):
        data = np.loadtxt(filename)
    else:
        print(f"Файл {filename} не найден. Использую примерные данные (вкл. p=16).")
        if default_data is None:
            # Дефолты с p=16 (ориентировочно из лекции/теста)
            default_data = np.array([[1, 10.2], [2, 6.5], [4, 5.3], [8, 4.8], [16, 4.5]])  # Для v1
        data = default_data
    p = data[:, 0].astype(int)
    times = data[:, 1]
    return p, times

# Примерные данные (включая p=16)
default_seq = np.array([[1, 10.2]])
default_par1 = np.array([[1, 10.2], [2, 6.5], [4, 5.3], [8, 4.8], [16, 4.5]])  # Scatterv: замедление на 16
default_par2 = np.array([[1, 10.2], [2, 5.8], [4, 3.8], [8, 2.5], [16, 2.0]])  # Sendrecv: лучше на 16

# Загрузка
p_seq, times_seq = load_times('times_seq.txt', default_seq)
p_par1, times_par1 = load_times('times_par1.txt', default_par1)
p_par2, times_par2 = load_times('times_par2.txt', default_par2)

# T1 для ускорения
T1 = times_seq[0]

# Ускорение S(p) = T1 / Tp
S_par1 = T1 / times_par1
S_par2 = T1 / times_par2

# Эффективность E(p) = S(p) / p * 100%
E_par1 = S_par1 / p_par1 * 100
E_par2 = S_par2 / p_par2 * 100

# График 1: Время выполнения (лог и линейная, как рис. 8.4-8.5)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.semilogy(p_par1, times_par1, 'b-o', label='Scatterv (v1)', markersize=6)
plt.semilogy(p_par2, times_par2, 'r-s', label='Sendrecv (v2)', markersize=6)
plt.xlabel('Число процессов p')
plt.ylabel('Время T(p), сек (лог-шкала)')
plt.title('Время выполнения')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(p_par1, times_par1, 'b-o', label='Scatterv (v1)')
plt.plot(p_par2, times_par2, 'r-s', label='Sendrecv (v2)')
plt.xlabel('Число процессов p')
plt.ylabel('Время T(p), сек (линейная)')
plt.title('Время выполнения')
plt.legend()
plt.grid(True)
plt.savefig('time_performance.png', dpi=150)
plt.show()

# График 2: Ускорение (как рис. 8.6, с p=16)
plt.figure(figsize=(8, 6))
p_max = max(max(p_par1), max(p_par2))
p_ideal = np.arange(1, p_max + 1)
plt.plot(p_ideal, p_ideal, 'k--', label='Идеальное S(p)=p', linewidth=2)
plt.plot(p_par1, S_par1, 'b-o', label='Scatterv (v1)', markersize=6)
plt.plot(p_par2, S_par2, 'r-s', label='Sendrecv (v2)', markersize=6)
plt.xlabel('Число процессов p')
plt.ylabel('Ускорение S(p) = T(1)/T(p)')
plt.title('Ускорение параллельных программ')
plt.legend()
plt.grid(True)
plt.savefig('acceleration.png', dpi=150)
plt.show()

# График 3: Эффективность (с p=16, падение на больших p)
plt.figure(figsize=(8, 6))
plt.plot(p_par1, E_par1, 'b-o', label='Scatterv (v1)', markersize=6)
plt.plot(p_par2, E_par2, 'r-s', label='Sendrecv (v2)', markersize=6)
plt.axhline(y=100, color='k', linestyle='--', label='Идеал E=100%')
plt.xlabel('Число процессов p')
plt.ylabel('Эффективность E(p) = S(p)/p × 100%, %')
plt.title('Эффективность параллелизма')
plt.legend()
plt.grid(True)
plt.savefig('efficiency.png', dpi=150)
plt.show()