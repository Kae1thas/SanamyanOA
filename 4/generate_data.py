# generate_data.py
import numpy as np

M = 10000  # строки матрицы A
N = 500    # столбцы матрицы A

print("Генерация aData.dat...")
a = np.arange(1, M+1, dtype=np.float64)
np.savetxt('aData.dat', a, fmt='%.6f')

print("Генерация AData.dat и xData.dat...")
A = np.random.rand(M, N)
x = np.random.rand(N)  # <-- ИСПРАВЛЕНО: N, а не M!

with open('in.dat', 'w') as f:
    f.write(f"{M} {N}\n")

np.savetxt('AData.dat', A, fmt='%.6f')
np.savetxt('xData.dat', x, fmt='%.6f')

print("Генерация bData.dat для CG...")
b = np.random.rand(M)
np.savetxt('bData.dat', b, fmt='%.6f')

print(f"Данные сгенерированы: A({M}x{N}), x({N}), b({M})")