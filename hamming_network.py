# Реализация сети Хэмминга для распознавания или классификации образов
# основной источник книга Головко В.А. Нейроинтеллект: Теория и применение
# Страница 234 Нейронная сеть Хэмминга
# Студент группы 121731
# Лабораторная работа №2 
# Вариант 6 Модель по сети Хэмминга
# Исамиддинов Ботир Бахтиер угли

# обозначения:
# n - размерность входного вектора (количество элементов в образце)
# m - количество эталонных образцов
# x - входной образ размерности n

# X - матрица эталонных образцов размера m x n
# d - расстояние Хэмминга между входным и эталонным образцом
# P - мера сходства между входным и эталонным образцом (P = n - d)
# w - веса первого слоя
# v - веса второго слоя (сети Хопфилда)
# T - пороговые значения нейронов


# ε (epsilon) - параметр торможения для сети Хопфилда (0 < ε < 1/m)
# y - выходы нейронов первого слоя
# z - выходы нейронов второго слоя

import numpy as np

class HammingNetwork:
    def __init__(self, reference_patterns, epsilon=0.5):
        """
        Инициализация сети Хэмминга
        
        Переменные:
        reference_patterns - матрица эталонных образцов X размера m x n
        n_patterns (m) - количество эталонных образцов
        n_inputs (n) - размерность входного вектора
        W1 (w) - матрица весов первого слоя
        T - вектор пороговых значений
        W2 (v) - матрица весов второго слоя
        epsilon (ε) - параметр торможения
        """
        self.reference_patterns = np.array(reference_patterns)
        self.n_patterns = len(reference_patterns)  # m
        self.n_inputs = len(reference_patterns[0])  # n
        
        # Веса первого слоя (формула 5.85): wij = xij/2
        # где xij - j-й элемент i-го эталонного образца
        self.W1 = self.reference_patterns / 2
        
        # Пороговые значения (формула 5.85): Tj = n/2
        # где n - размерность входного вектора
        self.T = np.ones(self.n_patterns) * self.n_inputs / 2
        
        # Веса второго слоя (формула 5.88):
        # vkj = 1, если k = j
        # vkj = -ε, если k ≠ j
        self.epsilon = epsilon
        self.W2 = np.eye(self.n_patterns) - epsilon * (1 - np.eye(self.n_patterns))
    
    def activation_function(self, x):
        """
        Пороговая функция активации (формула 5.91)
        F(Sj) = Sj, если Sj > 0
        F(Sj) = 0, если Sj ≤ 0
        
        Параметры:
        x - входной сигнал нейрона
        """
        return np.where(x > 0, x, 0)
    
    def forward_first_layer(self, input_pattern):
        """
        Прямое распространение через первый слой (формула 5.86)
        yj = Σ(wij*xi) + Tj = Σ(xij/2 * xi) + n/2
        
        Параметры:
        input_pattern (x) - входной образ размерности n
        
        Возвращает:
        y - выходы нейронов первого слоя, равные мере сходства P
        """
        return np.dot(self.W1, input_pattern) + self.T
    
    def forward_second_layer(self, y, max_iterations=100):
        """
        Итеративный процесс во втором слое (формула 5.92)
        zj(t) = F(zj(t-1) - ε*Σ(zk(t-1)))
        
        Параметры:
        y - выходы первого слоя (начальное состояние z(0))
        max_iterations - максимальное число итераций
        
        Возвращает:
        z - установившиеся выходы второго слоя
        """
        z = y.copy()
        
        for _ in range(max_iterations):
            z_old = z.copy()
            # Σ(zk(t-1)) для k≠j - сумма выходов всех нейронов, кроме j-го
            for j in range(self.n_patterns):
                inhibitory_sum = np.sum(z_old) - z_old[j]
                z[j] = self.activation_function(z_old[j] - self.epsilon * inhibitory_sum)
            
            if np.array_equal(z, z_old):
                break
        
        return z
    
    def recognize(self, input_pattern):
        """
        Распознавание входного образца
        
        Параметры:
        input_pattern (x) - входной образ
        
        Возвращает:
        recognized_pattern - распознанный эталонный образец
        pattern_index - индекс распознанного образца
        similarity (P) - мера сходства (n - d), где d - расстояние Хэмминга
        """
        # y - мера сходства входного образа с каждым эталонным
        y = self.forward_first_layer(input_pattern)
        
        # z - результат конкурентного процесса
        z = self.forward_second_layer(y)
        
        # Индекс нейрона-победителя
        winner_idx = np.argmax(z)
        
        # P = n - d, где d - расстояние Хэмминга
        similarity = y[winner_idx]
        
        return self.reference_patterns[winner_idx], winner_idx, similarity

