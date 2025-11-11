# cartesian_topology.py
from mpi4py import MPI
import numpy as np
import math

# Инициализация MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Проверка, является ли число процессов квадратом
root = int(math.sqrt(size))
if root * root != size:
    if rank == 0:
        print("Ошибка: Число процессов должно быть квадратом натурального числа!")
    MPI.Finalize()
    exit()

# Создание двумерной декартовой топологии (тор)
dims = (root, root)
periods = (True, True)
comm_cart = comm.Create_cart(dims=dims, periods=periods, reorder=True)

# Получение нового ранга и координат в топологии
new_rank = comm_cart.Get_rank()
coords = comm_cart.Get_coords(new_rank)

# Вывод нового ранга и координат
print(f"Процесс {rank} -> Новый ранг {new_rank}, Координаты {coords}")

# Этап 1.2: Определение соседей
neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)

# Вывод информации о соседях
print(f"Процесс {new_rank} (coords {coords}): Верхний сосед = {neighbour_up}, "
      f"Нижний сосед = {neighbour_down}, Левый сосед = {neighbour_left}, "
      f"Правый сосед = {neighbour_right}")

# Этап 1.3: Кольцевой обмен по горизонтали
# Создаём массив с уникальными данными (ранг процесса)
a = np.array([new_rank], dtype=np.float64)
total_sum = a.copy()

# Создаём подкоммуникатор для горизонтальной строки
row_comm = comm_cart.Sub((False, True))

# Собираем ранги соседей через Sendrecv_replace
for _ in range(dims[1] - 1):  # Пройти dims[1] - 1 раз
    send_buf = a.copy()  # Отправляем исходный ранг
    comm_cart.Sendrecv_replace(send_buf, dest=neighbour_right, source=neighbour_left)
    total_sum += send_buf

# Суммируем ранги в пределах строки
row_sum = np.array([new_rank], dtype=np.float64)  # Локальный ранг
row_sum = row_comm.allreduce(row_sum, op=MPI.SUM)

# Вывод суммы после кольцевого обмена
print(f"Процесс {new_rank} (coords {coords}): Сумма после кольца = {row_sum[0]}")