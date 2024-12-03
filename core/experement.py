from random import uniform
from numpy.random import normal
import numpy as np
from core.algorithms import Algorithms


def braunschweig(compos: np.ndarray) -> float:
    """Calculates the Braunschweig index for a given composition.

    Args:
        compos (np.ndarray): A 1D numpy array representing the composition (K, Na, N).

    Returns:
        float: The Braunschweig index.
    """
    K, Na, N = compos
    return (0.12 * (K + Na) + 0.24 * N + 0.48) / 100


def get_inorganic(prod_val: np.ndarray, compos_inorganic: np.ndarray) -> np.ndarray:
    """Subtracts the inorganic nutrients based on the Braunschweig index.

    Args:
        prod_val (np.ndarray): A 2D numpy array representing the production values.
        compos_inorganic (np.ndarray): A 2D numpy array representing the inorganic composition for each variety.

    Returns:
        np.ndarray: The updated production values after subtracting inorganic nutrients.
    """
    new_prod_val = prod_val.copy()  # Create a copy to avoid modifying the original array

    for variety in range(compos_inorganic.shape[0]):
        braun_res = braunschweig(compos_inorganic[variety])
        new_prod_val[:, variety] = np.maximum(0, new_prod_val[:, variety] - braun_res)  # Vectorized subtraction and clipping

    return new_prod_val


def generate_uniform_matrix(n: int, min_a: float, max_a: float, min_b: float, max_b: float) -> np.ndarray:
    """Generates a matrix with uniform random values.

    Args:
        n (int): The size of the matrix (n x n).
        min_a (float): Minimum value for the first row.
        max_a (float): Maximum value for the first row.
        min_b (float): Minimum multiplier for subsequent rows.
        max_b (float): Maximum multiplier for subsequent rows.

    Returns:
        np.ndarray: The generated matrix.
    """
    matrix = np.zeros((n, n))
    matrix[0] = np.round(np.random.uniform(min_a, max_a, n), 3)  # Vectorized first row generation

    for j in range(1, n):
        matrix[j] = np.round(matrix[j - 1] * np.random.uniform(min_b, max_b, n), 3) # Vectorized subsequent row generation

    return matrix


def generate_normal_matrix(n: int, min_a: float, max_a: float, avg: float, deviation: float) -> np.ndarray:
    """Generates a matrix with normally distributed random values.

    Args:
        n (int): The size of the matrix (n x n).
        min_a (float): Minimum value for the first row.
        max_a (float): Maximum value for the first row.
        avg (float): Mean of the normal distribution.
        deviation (float): Standard deviation of the normal distribution.

    Returns:
        np.ndarray: The generated matrix.
    """

    matrix = np.zeros((n, n))
    matrix[0] = np.round(np.random.uniform(min_a, max_a, n), 3)  # Vectorized first row generation
    normal_row = np.random.normal(avg, deviation, n)
    normal_row = np.clip(normal_row, 0.001, 0.999)

    for j in range(1, n):
        matrix[j] = np.round(matrix[j - 1] * normal_row, 3) # Simplified calculation

    return matrix



def generate_inorganic_matrix(n: int) -> np.ndarray:
    """Generates a matrix with inorganic composition values.

    Args:
        n (int): The number of rows in the matrix.

    Returns:
        np.ndarray: The generated matrix.
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
    """Runs the experiment with specified parameters.

    Args:
        n (int): Size of the matrix.
        t (int): Number of trials.
        min_a (float): Minimum initial value.
        max_a (float): Maximum initial value.
        min_b (float): Minimum multiplier.
        max_b (float): Maximum multiplier.
        consider_inorganic (bool): Whether to consider inorganic nutrients.
        is_normal (bool): Whether to use normal distribution.

    Returns:
        Algorithms: The Algorithms object with results.
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
