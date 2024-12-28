import numpy as np
from hamming_network import HammingNetwork

def print_pattern(pattern, width=None):
    """Выводит бинарный образ в виде символьной матрицы"""
    if width is None:
        width = int(np.sqrt(len(pattern)))
    
    for i in range(0, len(pattern), width):
        row = pattern[i:i+width]
        print(" ".join("#" if x == 1 else "." for x in row))
    print()

def add_noise(pattern, noise_level=0.1):
    """Добавляет случайный шум к образцу"""
    noisy_pattern = np.array(pattern).copy()  # Преобразуем в numpy массив
    n_pixels = int(len(pattern) * noise_level)
    indices = np.random.choice(len(pattern), n_pixels, replace=False)
    for idx in indices:  # Инвертируем каждый бит по отдельности
        noisy_pattern[idx] = 1 - noisy_pattern[idx]
    return noisy_pattern

# Создаем эталонные образы (цифры 0-9)
reference_patterns = [
    # Цифра 0
    [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,1,
     1,0,0,0,1,
     0,1,1,1,0],
    
    # Цифра 1
    [0,0,1,0,0,
     0,1,1,0,0,
     0,0,1,0,0,
     0,0,1,0,0,
     0,1,1,1,0],
     
    # Цифра 2
    [0,1,1,1,0,
     1,0,0,0,1,
     0,0,1,1,0,
     0,1,0,0,0,
     1,1,1,1,1]
]

# Создаем сеть Хэмминга
network = HammingNetwork(reference_patterns)

# Тестируем сеть
with open('results.txt', 'w', encoding='utf-8') as f:
    # Проверяем распознавание эталонных образцов
    f.write("=== Тест на эталонных образцах ===\n\n")
    for i, pattern in enumerate(reference_patterns):
        f.write(f"Эталонный образец {i}:\n")
        for j in range(0, len(pattern), 5):
            row = pattern[j:j+5]
            f.write(" ".join("#" if x == 1 else "." for x in row) + "\n")
        
        recognized, idx, similarity = network.recognize(pattern)
        f.write(f"\nРаспознан как образец {idx} с мерой сходства {similarity:.2f}\n")
        f.write("-" * 40 + "\n\n")
    
    # Проверяем распознавание зашумленных образцов
    f.write("=== Тест на зашумленных образцах ===\n\n")
    for i, pattern in enumerate(reference_patterns):
        noisy_pattern = add_noise(pattern, 0.20)  # 20% шума
        
        f.write(f"Зашумленный образец {i}:\n")
        for j in range(0, len(noisy_pattern), 5):
            row = noisy_pattern[j:j+5]
            f.write(" ".join("#" if x == 1 else "." for x in row) + "\n")
        
        recognized, idx, similarity = network.recognize(noisy_pattern)
        f.write(f"\nРаспознан как образец {idx} с мерой сходства {similarity:.2f}\n")
        
        f.write("\nВосстановленный образец:\n")
        for j in range(0, len(recognized), 5):
            row = recognized[j:j+5]
            f.write(" ".join("#" if x == 1 else "." for x in row) + "\n")
        f.write("-" * 40 + "\n\n")

        # Также выводим на экран
        print(f"Тестирование образца {i}:")
        print("Исходный образец:")
        print_pattern(pattern, 5)
        print("Зашумленный образец:")
        print_pattern(noisy_pattern, 5)
        print("Распознанный образец:")
        print_pattern(recognized, 5)
        print(f"Мера сходства: {similarity:.2f}")
        print("-" * 40)

