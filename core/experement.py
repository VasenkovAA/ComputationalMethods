import numpy as np
from core.algorithms import Algorithms


def braunschweig(compos: np.ndarray) -> float:
    """Вычисляет индекс Брауншвейга для заданного состава.

    Аргументы:
        compos (np.ndarray): Одномерный массив numpy, представляющий состав (K, Na, N).

    Возвращает:
        float: Индекс Брауншвейга.
    """
    K, Na, N = compos
    return (0.12 * (K + Na) + 0.24 * N + 0.48) / 100


def get_inorganic(prod_val: np.ndarray, compos_inorganic: np.ndarray) -> np.ndarray:
    """Вычитает неорганические питательные вещества на основе индекса Брауншвейга.

    Аргументы:
        prod_val (np.ndarray): Двумерный массив numpy, представляющий значения производства.
        compos_inorganic (np.ndarray): Двумерный массив numpy, представляющий неорганический состав для каждой разновидности.

    Возвращает:
        np.ndarray: Обновленные значения производства после вычитания неорганических питательных веществ.
    """
    new_prod_val = prod_val.copy()  # Create a copy to avoid modifying the original array

    for variety in range(compos_inorganic.shape[0]):
        braun_res = braunschweig(compos_inorganic[variety])
        new_prod_val[:, variety] = np.maximum(0, new_prod_val[:, variety] - braun_res)  # Vectorized subtraction and clipping

    return new_prod_val


def generate_uniform_matrix(n: int, min_a: float, max_a: float, min_b: float, max_b: float) -> np.ndarray:
    """Генерирует матрицу с равномерно распределенными случайными значениями.

    Аргументы:
        n (int): Размер матрицы (n x n).
        min_a (float): Минимальное значение для первой строки.
        max_a (float): Максимальное значение для первой строки.
        min_b (float): Минимальный множитель для последующих строк.
        max_b (float): Максимальный множитель для последующих строк.

    Возвращает:
        np.ndarray: Сгенерированная матрица.
    """
    matrix = np.zeros((n, n))
    matrix[0] = np.round(np.random.uniform(min_a, max_a, n), 3)  # Vectorized first row generation

    for j in range(1, n):
        matrix[j] = np.round(matrix[j - 1] * np.random.uniform(min_b, max_b, n), 3) # Vectorized subsequent row generation

    return matrix


def generate_normal_matrix(n: int, min_a: float, max_a: float, avg: float, deviation: float) -> np.ndarray:
    """Генерирует матрицу с нормально распределенными случайными значениями.

    Аргументы:
        n (int): Размер матрицы (n x n).
        min_a (float): Минимальное значение для первой строки.
        max_a (float): Максимальное значение для первой строки.
        avg (float): Среднее значение нормального распределения.
        deviation (float): Стандартное отклонение нормального распределения.

    Возвращает:
        np.ndarray: Сгенерированная матрица.
    """

    matrix = np.zeros((n, n))
    matrix[0] = np.round(np.random.uniform(min_a, max_a, n), 3)  # Vectorized first row generation
    normal_row = np.random.normal(avg, deviation, n)
    normal_row = np.clip(normal_row, 0.001, 0.999)

    for j in range(1, n):
        matrix[j] = np.round(matrix[j - 1] * normal_row, 3) # Simplified calculation

    return matrix



def generate_inorganic_matrix(n: int) -> np.ndarray:
    """Генерирует матрицу с неорганическими значениями состава.

    Аргументы:
        n (int): Количество строк в матрице.

    Возвращает:
        np.ndarray: Сгенерированная матрица.
    """
    min_K, max_K = 4, 8.7
    min_Na, max_Na = 0.15, 0.92
    min_N, max_N = 1.2, 3

    matrix = np.zeros((n, 3))
    matrix[:, 0] = np.round(np.random.uniform(min_K, max_K, n), 3) # Vectorized K generation
    matrix[:, 1] = np.round(np.random.uniform(min_Na, max_Na, n), 3) # Vectorized Na generation
    matrix[:, 2] = np.round(np.random.uniform(min_N, max_N, n), 3) # Vectorized N generation

    return matrix


def experiment(n: int, t: int, min_a: float, max_a: float, min_b: float, max_b: float,
              consider_inorganic: bool, is_normal: bool) -> Algorithms:
    """Запускает эксперимент с заданными параметрами.

    Аргументы:
        n (int): Размер матрицы.
        t (int): Количество испытаний.
        min_a (float): Минимальное начальное значение.
        max_a (float): Максимальное начальное значение.
        min_b (float): Минимальный множитель.
        max_b (float): Максимальный множитель.
        consider_inorganic (bool): Нужно ли учитывать неорганические питательные вещества.
        is_normal (bool): Нужно ли использовать нормальное распределение.

    Возвращает:
        Algorithms: Объект Algorithms с результатами.
    """
    algorithms = Algorithms(n)

    for _ in range(t):
        P = generate_normal_matrix(n, min_a, max_a, min_b, max_b) if is_normal else generate_uniform_matrix(n, min_a, max_a, min_b, max_b)
        if consider_inorganic:
            inorganic_matrix = generate_inorganic_matrix(n)
            P = get_inorganic(P, inorganic_matrix)
        algorithms.run_algorithms(P)

    algorithms.calculate_average(t)
    return algorithms
