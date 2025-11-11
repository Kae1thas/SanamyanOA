import matplotlib.pyplot as plt

# Времена из matrix_vector.py
t_seq_matrix = 0.016744  # Среднее из 0.014330 и 0.019157
t_par_matrix_2 = 0.000413  # 2 процесса
t_par_matrix_4 = 0.000353  # 4 процесса

# Времена из dot_product.py
t_seq_dot = 0.0000465  # Среднее из 0.000058 и 0.000035
t_par_dot_reduce_2 = 0.000109  # 2 процесса, Reduce
t_par_dot_reduce_4 = 0.000125  # 4 процесса, Reduce
t_par_dot_allreduce_2 = 0.00003  # 2 процесса, Allreduce (усреднено)
t_par_dot_allreduce_4 = 0.00004425  # 4 процесса, Allreduce (усреднено)

# Число процессов
processes = [2, 4]

# Времена для каждого варианта (только параллельные)
times_matrix = [t_par_matrix_2, t_par_matrix_4]
times_dot_reduce = [t_par_dot_reduce_2, t_par_dot_reduce_4]
times_dot_allreduce = [t_par_dot_allreduce_2, t_par_dot_allreduce_4]

# Ускорение для каждого варианта
speedup_matrix = [t_seq_matrix / t for t in times_matrix]
speedup_dot_reduce = [t_seq_dot / t for t in times_dot_reduce]
speedup_dot_allreduce = [t_seq_dot / t for t in times_dot_allreduce]

# Построение графика
plt.plot(processes, speedup_matrix, marker='o', label='Matrix-Vector')
plt.plot(processes, speedup_dot_reduce, marker='o', label='Dot Product (Reduce)')
plt.plot(processes, speedup_dot_allreduce, marker='o', label='Dot Product (Allreduce)')
plt.xlabel('Число процессов')
plt.ylabel('Ускорение')
plt.title('Ускорение в зависимости от числа процессов')
plt.grid(True)
plt.legend()
plt.savefig('speedup_plot.png')