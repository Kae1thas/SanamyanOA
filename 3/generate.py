import numpy as np

def write_matrix(A, filename):
    with open(filename, 'w') as f:
        for j in range(A.shape[0]):
            for i in range(A.shape[1]):
                f.write(f"{A[j,i]:.12e}\n")

def write_vector(v, filename):
    with open(filename, 'w') as f:
        for x in v:
            f.write(f"{x:.12e}\n")

if __name__ == "__main__":
    N = 500   
    M = 1000  
    np.random.seed(42)
    U = np.random.randn(M, N)
    A = U
    x_true = np.random.randn(N)
    b = A.dot(x_true) + 1e-4 * np.random.randn(M) 

    with open('in.dat','w') as f:
        f.write(f"{N}\n{M}\n")
    write_matrix(A, 'AData.dat')
    write_vector(b, 'bData.dat')
    write_vector(x_true, 'xTrue.dat')
    print("Сгенерировано: in.dat, AData.dat, bData.dat, xTrue.dat")
