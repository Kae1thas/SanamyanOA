import numpy as np
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')

start_time = time.time()
with open('in.dat', 'r', encoding='utf-8') as f:
    M, N = map(int, f.read().split())
A = np.loadtxt('AData.dat').reshape(M, N)
x = np.loadtxt('xData.dat')
b = np.dot(A, x)
np.savetxt('Results_seq.dat', b, fmt='%.6f')
print(f"Последовательная версия: {time.time() - start_time} секунд")