import numpy as np
N, M = 100, 80 
np.savetxt('in.dat', [N, M], fmt='%d')
B = np.random.rand(M, N)
A = B.T @ B  
np.savetxt('AData.dat', A.flatten(), fmt='%.6f')
x_true = np.random.rand(N)
b = A @ x_true
np.savetxt('bData.dat', b, fmt='%.6f')
print('SPD-данные созданы! N=', N, 'M=', M, 'строк в AData.dat:', M*N)