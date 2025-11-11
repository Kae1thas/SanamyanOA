import numpy as np

# Последовательный метод прогонки
def sequential_thomas(a, b, c, d):
    n = len(b)
    c_prime = np.zeros(n-1)  # Вспомогательный массив для верхней диагонали
    d_prime = np.zeros(n)    # Вспомогательный массив для правой части
    
    # Прямой ход
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    # Обратный ход
    x = np.zeros(n)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x