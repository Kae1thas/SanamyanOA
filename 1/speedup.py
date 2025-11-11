import matplotlib.pyplot as plt

# Времена из sequential.py
t_seq = 0.43462

# Времена из parallel_send_recv.py (Send/Recv)
t_par2 = 0.40914   # 2 процесса
t_par4 = 0.39883   # 4 процесса
t_par8 = 0.49825   # 8 процессов

# Времена из parallel_scatter_gather.py (Scatterv/Gatherv)
t_scatter2 = 0.42777  
t_scatter4 = 0.41500
t_scatter8 = 0.45729

# Времена из parallel_scatter_gather_variable.py (Переменный M)
t_variable2 = 0.41581  
t_variable4 = 0.42812
t_variable8 = 0.44101

# Число процессов
processes = [1, 2, 4, 8]

# Времена для каждого варианта
times_send_recv = [t_seq, t_par2, t_par4, t_par8]
times_scatter = [t_seq, t_scatter2, t_scatter4, t_scatter8]
times_variable = [t_seq, t_variable2, t_variable4, t_variable8]

# Ускорение для каждого варианта
speedup_send_recv = [times_send_recv[0] / t for t in times_send_recv]
speedup_scatter = [times_scatter[0] / t for t in times_scatter]
speedup_variable = [times_variable[0] / t for t in times_variable]

# Построение графика
plt.plot(processes, speedup_send_recv, marker='o', label='Send/Recv')
plt.plot(processes, speedup_scatter, marker='o', label='Scatterv/Gatherv')
plt.plot(processes, speedup_variable, marker='o', label='Variable M')
plt.xlabel('Number of processes')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of Processes')
plt.grid(True)
plt.legend()
plt.savefig('speedup_plot.png')
