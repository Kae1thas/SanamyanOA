# run_benchmarks.py
import subprocess
import os
import time
import numpy as np
from datetime import datetime

# Параметры
M = 10000
N = 500
PROCESSES = [2, 4, 8, 16, 32, 64]
PROGRAMS = {
    "matrix_vector": "matrix_vector.py",
    "scatter_gather": "parallel_scatter_gather.py",
    "scatter_gather_var": "parallel_scatter_gather_variable.py",
    "cg_simple": "cg_simple.py",
    "cg_parallel": "parallel_cg.py",
}

# Создание данных
def generate_data():
    print("Генерация данных...")
    subprocess.run(["python3", "generate_data.py"], check=True)
    # Генерация bData.dat для CG
    b = np.random.rand(M)
    np.savetxt('bData.dat', b, fmt='%.6f')

# Последовательная версия (для базового времени)
def run_sequential():
    print("Запуск последовательной версии...")
    result = subprocess.run(["python3", "sequential.py"], capture_output=True, text=True)
    
    # Ищем строку: "Последовательная версия: X.XXXXXX секунд"
    for line in result.stdout.splitlines():
        if "Последовательная версия" in line and "секунд" in line:
            try:
                seq_time = float(line.split()[-2])
                print(f"  Найдено время: {seq_time:.6f} с")
                return seq_time
            except:
                pass
    
    # Если не нашли — попробуем вручную запустить и измерить
    print("  Время не найдено в выводе. Измеряю вручную...")
    t0 = time.time()
    subprocess.run(["python3", "sequential.py"], check=True)
    t1 = time.time()
    seq_time = t1 - t0
    print(f"  Измерено вручную: {seq_time:.6f} с")
    return seq_time

# Запуск одной программы
def run_mpi(program, np):
    print(f"  Запуск {program} на {np} процессах...")
    cmd = ["mpiexec", "-n", str(np), "--oversubscribe", "python3", PROGRAMS[program]]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"  ТАЙМАУТ: {program} на {np} процессах")
        return None

    if result.returncode != 0:
        print(f"  ОШИБКА: {program} завершился с кодом {result.returncode}")
        print(result.stderr)
        return None

    # ИЩЕМ СТРОКУ: TIME_ELAPSED: X.XXXXXX
    time_val = None
    for line in result.stdout.splitlines():
        if "TIME_ELAPSED:" in line:
            try:
                time_val = float(line.split()[1])
                break
            except:
                pass

    if time_val is None:
        print(f"  Время не найдено в выводе {program}")
        print("  Вывод:")
        print(result.stdout)
        return None

    print(f"  Время: {time_val:.6f} с")
    return time_val

# Основной цикл
def main():
    os.makedirs("results", exist_ok=True)
    generate_data()
    seq_time = run_sequential()
    
    with open("results/times.csv", "w") as f:
        f.write("program,procs,time,speedup,efficiency\n")
        f.write(f"sequential,1,{seq_time:.6f},1.0,1.0\n")
        
        for prog in PROGRAMS.keys():
            for np in PROCESSES:
                t = run_mpi(prog, np)
                if t is not None:
                    speedup = seq_time / t
                    efficiency = speedup / np
                    f.write(f"{prog},{np},{t:.6f},{speedup:.6f},{efficiency:.6f}\n")
                else:
                    f.write(f"{prog},{np},ERROR,,\n")
                time.sleep(1)  # Чтобы не перегружать систему

if __name__ == "__main__":
    main()