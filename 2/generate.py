import numpy as np

M, N = 50, 50  
A = np.random.rand(M, N) 
x = np.random.rand(M)     

with open('in.dat', 'w') as f:
    f.write(f"{M}\n{N}\n")  

np.savetxt('AData.dat', A, fmt='%.6f')  
np.savetxt('xData.dat', x, fmt='%.6f')  