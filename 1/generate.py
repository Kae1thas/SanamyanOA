import numpy as np
M, N = 1000, 1000
A = np.random.rand(M, N) 
x = np.random.rand(N)
with open('in.dat', 'w') as f:
    f.write(f"{M} {N}\n")
np.savetxt('AData.dat', A, fmt='%.6f')
np.savetxt('xData.dat', x.reshape(1, -1), fmt='%.6f', delimiter=' ')