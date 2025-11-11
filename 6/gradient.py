from mpi4py import MPI
import numpy as np
import math

def matrix_vector_multiply(A_part, x, comm_cart, neighbour_left, neighbour_right, n_local, n):
    Ax_local = np.dot(A_part, x) 
    
    send_buf = Ax_local.copy()
    total_Ax = Ax_local.copy()
    for _ in range(comm_cart.dims[1] - 1):
        comm_cart.Sendrecv_replace(send_buf, dest=neighbour_right, source=neighbour_left)
        total_Ax += send_buf
    
    return total_Ax

def scalar_product(v_local, w_local, comm_cart, neighbour_left, neighbour_right, n_local):
    local_dot = np.dot(v_local, w_local)
    
    if not np.isfinite(local_dot):
        local_dot = 0.0
    
    send_buf = np.array([local_dot], dtype=np.float64)
    total_dot = send_buf.copy()
    for _ in range(comm_cart.dims[1] - 1):
        comm_cart.Sendrecv_replace(send_buf, dest=neighbour_right, source=neighbour_left)
        total_dot += send_buf
        if not np.isfinite(total_dot):
            total_dot = np.array([0.0], dtype=np.float64)
    
    return total_dot[0]

def conjugate_gradient_method(A_part, b_local, x, comm_cart, n_local, n, max_iter=1000, tol=1e-4):

    neighbour_up, neighbour_down = comm_cart.Shift(0, 1)
    neighbour_left, neighbour_right = comm_cart.Shift(1, 1)
    rank = comm_cart.Get_rank()
    coords = comm_cart.Get_coords(rank)
    

    r_local = b_local - matrix_vector_multiply(A_part, x, comm_cart, neighbour_left, neighbour_right, n_local, n)
    p = r_local.copy()  
    
    p_full = np.zeros(n, dtype=np.float64)
    start_idx = coords[0] * n_local
    p_full[start_idx:start_idx + n_local] = p
    comm_cart.Allreduce(MPI.IN_PLACE, p_full, op=MPI.SUM)
    
    rsold = scalar_product(r_local, r_local, comm_cart, neighbour_left, neighbour_right, n_local)
    if rank == 0:
        print(f"Итерация 0: rsold = {rsold}")
    
    for iter in range(max_iter):
        Ap_local = matrix_vector_multiply(A_part, p_full, comm_cart, neighbour_left, neighbour_right, n_local, n)
        pAp = scalar_product(p, Ap_local, comm_cart, neighbour_left, neighbour_right, n_local)
        
        if abs(pAp) < 1e-10:
            if rank == 0:
                print(f"Итерация {iter}: pAp = {pAp}, деление на ноль, останавливаем")
            break
        
        alpha = rsold / pAp
        if not np.isfinite(alpha):
            if rank == 0:
                print(f"Итерация {iter}: alpha = {alpha}, останавливаем")
            break
        
        x += alpha * p_full
        r_local -= alpha * Ap_local
        
        rsnew = scalar_product(r_local, r_local, comm_cart, neighbour_left, neighbour_right, n_local)
        if rank == 0:
            print(f"Итерация {iter + 1}: rsnew = {rsnew}, alpha = {alpha}")
        
        if not np.isfinite(rsnew) or np.sqrt(rsnew) < tol:
            break
            
        p = r_local + (rsnew / rsold) * p
        p_full[start_idx:start_idx + n_local] = p
        comm_cart.Allreduce(MPI.IN_PLACE, p_full, op=MPI.SUM)
        rsold = rsnew
    
    return x, iter + 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

root = int(math.sqrt(size))
if root * root != size:
    if rank == 0:
        print("Ошибка: Число процессов должно быть квадратом!")
    MPI.Finalize()
    exit()

dims = (root, root)
periods = (True, True)
comm_cart = comm.Create_cart(dims=dims, periods=periods, reorder=True)
coords = comm_cart.Get_coords(comm_cart.Get_rank())

n = 100  
n_local = n // root 

np.random.seed(42)  
A_full = np.random.rand(n, n)
A_full = (A_full + A_full.T) / 2  
A_full += 100 * n * np.eye(n) 
start_idx = coords[0] * n_local
A_part = A_full[start_idx:start_idx + n_local, :]  
b_local = np.random.rand(n_local)  
x = np.zeros(n) 

start_time = MPI.Wtime()

x, iterations = conjugate_gradient_method(A_part, b_local, x, comm_cart, n_local, n)

end_time = MPI.Wtime()

if comm_cart.Get_rank() == 0:
    print(f"Процесс {comm_cart.Get_rank()} (coords {coords}): Итераций = {iterations}, Время = {end_time - start_time:.4f} сек")