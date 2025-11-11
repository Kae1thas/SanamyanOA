#!/bin/bash

# Массив с количествами процессов
PROCS=(1 4 9 16)

# Лог-файл для результатов
LOG_FILE="benchmarks.log"
echo "=== Бенчмарки для gradient.py ===" > $LOG_FILE
echo "Дата: $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

echo "Запуск бенчмарков для процессов: ${PROCS[@]}"
echo "Результаты сохраняются в $LOG_FILE"

for n in "${PROCS[@]}"; do
    echo "----------------------------------------"
    echo "Запуск с $n процессами..."
    
    # Запуск mpiexec с time: %e - реальное время в секундах
    /usr/bin/time -f "Общее время выполнения: %e сек" \
        mpiexec -n $n --oversubscribe python3 gradient.py 2>&1 | tee temp_output.txt
    
    # Захват вывода (включая время из gradient.py)
    OUTPUT=$(cat temp_output.txt)
    
    # Поиск строки с временем из gradient.py (предполагаем, что она содержит "{end_time - start_time:.4f}")
    TIME_LINE=$(echo "$OUTPUT" | grep -E "\{[0-9]+\.[0-9]{4}\}")
    
    echo "Вывод с $n процессами:"
    echo "$OUTPUT"
    
    # Сохранение в лог
    echo "=== $n процессов ===" >> $LOG_FILE
    echo "$OUTPUT" >> $LOG_FILE
    if [ -n "$TIME_LINE" ]; then
        echo "Время из gradient.py: $TIME_LINE" >> $LOG_FILE
    fi
    echo "" >> $LOG_FILE
    
    # Удаляем временный файл
    rm -f temp_output.txt
    
    echo "Готово для $n процессов."
done

echo "----------------------------------------"
echo "Все бенчмарки завершены. Лог: $LOG_FILE"