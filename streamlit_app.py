import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

# Установим заголовок приложения
st.title("Сеть Хопфилда для Распознавания Образов")

# Скрываем стандартный меню и футер Streamlit для минималистичного вида
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Функция для генерации случайного тестового образа
def generate_random_test_pattern():
    """
    Генерирует случайный тестовый образ с значениями 1, -1 и 0.

    Возвращает:
        np.array: Случайный тестовый образ.
    """
    return np.random.choice([1, -1, 0], size=25)


# Функция для инициализации весов сети Хопфилда методом Хебба
def initialize_weights(patterns):
    """
    Инициализирует веса сети Хопфилда на основе эталонных образцов методом Хебба.

    Аргументы:
        patterns (list of np.array): Список эталонных образцов.

    Возвращает:
        np.array: Матрица весов.
    """
    if not patterns:
        return np.zeros((25, 25))  # Возвращаем нулевую матрицу, если паттерны отсутствуют
    num_neurons = patterns[0].size  # Определяем количество нейронов
    weights = np.zeros((num_neurons, num_neurons))  # Инициализируем матрицу весов нулями
    for p in patterns:
        # Метод Хебба: W = W + p * p^T
        # Формула: W_ij = W_ij + p_i * p_j
        weights += np.outer(p, p)  # Внешнее произведение паттерна на себя
    np.fill_diagonal(weights, 0)  # Обнуляем диагональные элементы (нейрон не связан сам с собой)
    return weights / len(patterns)  # Нормализуем веса


# Функция активации
def activation_function(x):
    """
    Модифицированная функция знака для функции активации нейронов.

    Аргументы:
        x (float): Входное значение.

    Возвращает:
        int: 1, -1 или 0.
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# Асинхронное обновление состояния сети Хопфилда
def asynchronous_update(state, weights):
    """
    Выполняет асинхронное обновление состояния сети Хопфилда.

    Аргументы:
        state (np.array): Текущее состояние сети.
        weights (np.array): Матрица весов сети.

    Возвращает:
        tuple: Новое состояние сети и количество итераций.
    """
    num_neurons = len(state)  # Количество нейронов
    prev_state = np.copy(state)  # Сохраняем предыдущее состояние сети
    iteration = 0  # Счётчик итераций
    max_iterations = 1000  # Максимальное количество итераций для предотвращения зацикливания

    while iteration < max_iterations:
        iteration += 1
        neuron_indices = np.random.permutation(
            num_neurons)  # Случайным образом перемешиваем порядок обновления нейронов
        for i in neuron_indices:
            # Вычисляем взвешенную сумму входов для нейрона i: h_i = sum_j(W_ij * s_j)
            net_input = np.dot(weights[i], state)
            # Обновляем состояние нейрона с использованием функции активации: s_i = sign(h_i)
            state[i] = activation_function(net_input)

        # Проверяем, достигнута ли сходимость (нет изменений состояния сети)
        if np.array_equal(state, prev_state):
            break
        prev_state = np.copy(state)  # Обновляем предыдущее состояние сети

    return state, iteration  # Возвращаем новое состояние сети и количество итераций


# Функция для добавления шума к паттерну
def add_noise(pattern, noise_level):
    """
    Добавляет шум к образу путем случайного изменения части его элементов.

    Аргументы:
        pattern (np.array): Исходный образ.
        noise_level (float): Доля элементов, которые будут зашумлены (от 0 до 1).

    Возвращает:
        np.array: Зашумленный образ.
    """
    noisy_pattern = np.copy(pattern)  # Создаем копию исходного образа
    num_neurons = len(pattern)  # Количество нейронов
    num_noisy = int(noise_level * num_neurons)  # Количество нейронов, которые будут зашумлены
    if num_noisy == 0:
        return noisy_pattern  # Если уровень шума 0, возвращаем исходный образ
    noisy_indices = np.random.choice(num_neurons, num_noisy, replace=False)  # Выбираем случайные индексы для зашумления
    for idx in noisy_indices:
        noisy_pattern[idx] = 0  # Устанавливаем выбранные нейроны в состояние 0
    return noisy_pattern  # Возвращаем зашумленный образ


# Функция для отображения паттерна
def display_pattern(pattern, size=(5, 5)):
    """
    Отображает образ в виде матрицы с использованием символов.

    Аргументы:
        pattern (np.array): Образ для отображения.
        size (tuple): Размер матрицы (строки, столбцы).

    Возвращает:
        str: Строковое представление матрицы.
    """
    display = ""
    for i in range(size[0]):  # Проходим по строкам
        for j in range(size[1]):  # Проходим по столбцам
            idx = i * size[1] + j  # Вычисляем индекс нейрона
            if idx < len(pattern):
                if pattern[idx] == 1:
                    display += "● "  # Активный нейрон
                elif pattern[idx] == -1:
                    display += "○ "  # Неактивный нейрон
                else:
                    display += "△ "  # Зашумленный нейрон
            else:
                display += "  "  # Пустое место
        display += "\n"  # Переход на новую строку
    return display  # Возвращаем строковое представление образа


# Функция для подготовки данных для скачивания
def prepare_download_data(noisy_test, final_state, iterations, match, matched_pattern, similarity_scores):
    """
    Подготавливает данные для скачивания в формате CSV.

    Аргументы:
        noisy_test (np.array): Исходный тестовый образ с шумом.
        final_state (np.array): Результирующий образ после сходимости.
        iterations (int): Количество итераций до сходимости.
        match (bool): Совпадает ли образ с эталонным паттерном.
        matched_pattern (int): Номер совпавшего паттерна.
        similarity_scores (list): Меры сходства с каждым паттерном.

    Возвращает:
        str: Данные в формате CSV.
    """
    data = {
        'Neuron': [f'N{i + 1}' for i in range(25)],
        'Noisy Test': noisy_test,
        'Final State': final_state
    }
    df = pd.DataFrame(data)
    summary = {
        'Iterations': [iterations],
        'Match': [match],
        'Matched Pattern': [matched_pattern if match else 'None'],
        'Max Similarity': [len(final_state) if match else (max(similarity_scores) if similarity_scores else 0)]
    }
    df_summary = pd.DataFrame(summary)

    # Объединяем оба DataFrame
    combined = pd.concat([df, df_summary], axis=1)

    # Преобразуем в CSV
    csv = combined.to_csv(index=False)
    return csv


# Инициализация паттернов в session_state при первом запуске
if 'patterns' not in st.session_state:
    st.session_state['patterns'] = []

# Раздел для определения новых паттернов
st.header("Определение Эталонных Паттернов")

# Выбор количества паттернов для хранения
num_patterns = st.number_input(
    "Количество паттернов для хранения:",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

# Создание паттернов
for p in range(num_patterns):
    with st.expander(f"Паттерн {p + 1}"):
        pattern = []
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                for j in range(5):
                    idx = i * 5 + j
                    key = f"pattern_{p}_{idx}"
                    val = st.selectbox(
                        f"N{j + 1}",
                        options=[1, -1],
                        key=key,
                        index=0
                    )
                    pattern.append(val)
        pattern = np.array(pattern)
        if len(st.session_state['patterns']) > p:
            st.session_state['patterns'][p] = pattern
        else:
            st.session_state['patterns'].append(pattern)

# Инициализация весов сети после определения паттернов
weights = initialize_weights(st.session_state['patterns'])

# Раздел для тестирования сети
st.header("Тестирование Сети Хопфилда")

st.subheader("Ввод Тестового Образца")
st.markdown(
    "Введите значения нейронов для тестового образа. Используйте `1` для активных и `-1` для неактивных нейронов.")

# Создание сетки для ввода тестового образа
test_pattern = []
grid = st.columns(5)  # Создаем 5 столбцов для 5 нейронов в строке
for i in range(5):  # Проходим по строкам
    with grid[i]:
        row_pattern = []
        for j in range(5):  # Проходим по столбцам
            idx = i * 5 + j  # Вычисляем индекс нейрона
            key = f"test_{idx}"  # Уникальный ключ для каждого нейрона
            # Находим индекс текущего значения нейрона в списке [1, -1]
            try:
                index = [1, -1].index(st.session_state['test_pattern'][idx])
            except (KeyError, ValueError):
                index = 0  # Если значение не найдено, устанавливаем индекс по умолчанию
            val = st.selectbox(
                f"N{idx + 1}",
                options=[1, -1],
                key=key,
                index=index
            )  # Ввод значения нейрона
            # Обновляем значение в session_state при изменении
            if 'test_pattern' not in st.session_state:
                st.session_state['test_pattern'] = [1] * 25  # Инициализация, если отсутствует
            st.session_state['test_pattern'][idx] = val
            row_pattern.append(val)
        test_pattern.extend(row_pattern)  # Добавляем значения строки в общий тестовый образ
test_pattern = np.array(test_pattern)  # Преобразуем в numpy массив

# Кнопка для генерации нового случайного тестового образа
if st.button("Сгенерировать новый тестовый образ"):
    st.session_state['test_pattern'] = generate_random_test_pattern()
    st.experimental_rerun()  # Перезапускаем приложение для обновления значений

# Выбор уровня шума
noise_level = st.slider(
    "Уровень шума (доля зашумленных нейронов):",
    min_value=0.0,
    max_value=0.5,
    value=0.0,
    step=0.05
)

# Кнопка для запуска распознавания
if st.button("Распознать образ"):
    # Проверка наличия паттернов
    if not st.session_state['patterns']:
        st.error("Пожалуйста, определите хотя бы один эталонный паттерн перед распознаванием.")
    else:
        # Добавление шума к тестовому образу
        noisy_test = add_noise(test_pattern, noise_level)

        # Асинхронное обновление сети для распознавания
        final_state, iterations = asynchronous_update(noisy_test, weights)

        # Отображение результатов
        st.subheader("Результаты распознавания")
        cols_results = st.columns(2)  # Создаем два столбца для исходного и результирующего образов

        with cols_results[0]:
            st.markdown("**Исходный тестовый образ с шумом:**")
            st.text(display_pattern(noisy_test, size=(5, 5)))

        with cols_results[1]:
            st.markdown("**Результирующий образ после сходимости:**")
            st.text(display_pattern(final_state, size=(5, 5)))

        st.write(f"**Количество итераций до сходимости:** {iterations}")

        # Проверка, соответствует ли результат одному из эталонных паттернов
        match = False
        matched_pattern = None
        similarity_scores = []
        for idx, p in enumerate(st.session_state['patterns']):
            if np.array_equal(final_state, p):
                st.success(f"Образ совпадает с паттерном {idx + 1}.")
                match = True
                matched_pattern = idx + 1
                break
            else:
                similarity = np.sum(final_state == p)  # Мера сходства: количество совпадающих нейронов
                similarity_scores.append(similarity)
        if not match:
            # Если совпадения нет, рассчитаем меры сходства
            if not similarity_scores:
                similarity_scores = [0]  # Если нет паттернов, устанавливаем 0
            max_similarity = max(similarity_scores)
            st.warning("Образ не совпадает ни с одним из эталонных паттернов.")
            st.write(f"**Максимальная мера сходства:** {max_similarity}/{len(final_state)}")

        # Объяснение результата
        st.markdown("### Объяснение Результатов")
        if match:
            st.markdown(f"""
            **Сеть Хопфилда успешно распознала введенный образ как паттерн **{matched_pattern}**.

            Это произошло благодаря тому, что результирующий образ полностью совпадает с одним из обученных эталонных паттернов. 
            В процессе распознавания сеть использовала весовую матрицу, инициализированную методом Хебба, что позволило эффективно сохранить и воспроизвести паттерны.
            """)
        else:
            st.markdown(f"""
            **Сеть Хопфилда не смогла точно распознать введенный образ.**

            Однако максимальная мера сходства составляет **{max_similarity}/{len(final_state)}**, что указывает на частичное совпадение с одним из обученных эталонных паттернов. 
            Это может быть связано с уровнем шума или особенностями введенного образа, которые делают его менее похожим на сохраненные паттерны.
            """)

        # Подготовка данных для скачивания
        download_data = prepare_download_data(
            noisy_test,
            final_state,
            iterations,
            match,
            matched_pattern,
            similarity_scores
        )

        st.markdown("### Сохранение и Скачивание Результатов")
        st.markdown("""
        Вы можете сохранить результаты распознавания, включая исходный зашумленный образ, результирующий образ после сходимости, количество итераций, а также информацию о совпадении с эталонными паттернами.
        """)

        # Кнопка для скачивания результатов
        st.download_button(
            label="Скачать результаты в CSV",
            data=download_data,
            file_name='hopfield_results.csv',
            mime='text/csv',
        )

# Дополнительная информация
st.sidebar.header("Информация о сети Хопфилда")
st.sidebar.markdown("""
Сеть Хопфилда — это рекуррентная нейронная сеть, которая используется для хранения и извлечения ассоциативных воспоминаний. Она может быть применена для решения задач распознавания образов, ассоциативного поиска и оптимизационных задач.

**Особенности реализации:**
- **Асинхронное обновление:** Нейроны обновляются по одному за итерацию в случайном порядке, что имитирует более реалистичное поведение нейронов.
- **Функция активации:** Модифицированная функция знака, позволяющая использовать нейроны со значением `0` для зашумленных данных.
- **Метод Хебба:** Используется для инициализации весовой матрицы на основе эталонных паттернов.
- **Проверка сходимости:** Ограничение на максимальное количество итераций для предотвращения зацикливания.

**Структура данных:**
- **Эталонные паттерны:** Хранятся как список numpy массивов, где каждый массив представляет собой вектор состояний нейронов (`1` или `-1`).
- **Весовая матрица:** Матрица весов, инициализированная методом Хебба на основе эталонных паттернов.
- **Тестовый образ:** Вводится пользователем и может содержать зашумленные нейроны (`0`).

**Процесс работы сети:**
1. **Инициализация:** Пользователь определяет эталонные паттерны, вводя значения нейронов.
2. **Тестирование:** Пользователь вводит тестовый образ и выбирает уровень шума.
3. **Распознавание:** Сеть асинхронно обновляет состояния нейронов до достижения состояния релаксации или достижения максимального числа итераций.
4. **Результаты:** Отображаются исходный зашумленный образ, результирующий образ после сходимости, количество итераций и информация о совпадении с эталонными паттернами.

**Применение:**
- **Распознавание образов:** Восстановление исходных образов из зашумленных входных данных.
- **Ассоциативная память:** Поиск связанных образов на основе частичных или искажённых входных данных.
""")
