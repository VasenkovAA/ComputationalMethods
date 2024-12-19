import numpy as np
from scipy.optimize import linear_sum_assignment


class Algorithm:

    def __init__(self, name: str, func, num_days: int):
        self.name = name
        self.func = func
        self.ans: list[float] = [0.0] * num_days
        self.column_indexes: list[int] = []

    def __call__(self, matrix: np.matrix):
        return self.func(matrix)


class Algorithms:
    NUM_ALGORITHMS_FOR_ERROR = 7

    def __init__(self, num_days: int):
        self.__algorithms = [
            Algorithm('Max', self.__hungarian_max, num_days),
            Algorithm('Min', self.__hungarian_min, num_days),
            Algorithm('Greedy', self.__greedy_max, num_days),
            Algorithm('Thrifty', self.__greedy_min, num_days),
            Algorithm('GT', self.__greedy_thrifty, num_days),
            Algorithm('TG', self.__thrifty_greedy, num_days),
            Algorithm('T(k={})G', self.__TkG, num_days),
            Algorithm('G(k={})', self.__Gk, num_days),
            Algorithm('CTG', self.__CTG, num_days)
        ]
        self.num_days = num_days
        self.v = num_days // 2
        self.TkG = 0
        self.Gk = 0

    def __getitem__(self, item):
        return self.__algorithms[item]

    def __len__(self):
        return len(self.__algorithms)

    def __change_name_TkG(self, k: int):
        self.__algorithms[6].name = self.__algorithms[6].name.format(k)

    def __change_name_Gk(self, k: int):
        self.__algorithms[7].name = self.__algorithms[7].name.format(k)
    def run_algorithms(self, matrix: np.matrix):
        for algorithm in self.__algorithms:
            row_ind, col_ind = algorithm(matrix)
            for i in range(self.num_days):
                algorithm.ans[i] += matrix[row_ind[i], col_ind[i]]
            algorithm.column_indexes = col_ind

    def calculate_average(self, t: int):
        for algorithm in self.__algorithms:
            algorithm.ans[0] /= t
            for i in range(1, self.num_days):
                algorithm.ans[i] /= t
                algorithm.ans[i] += algorithm.ans[i - 1]

        self.TkG = round(self.TkG / t)
        self.__change_name_TkG(self.TkG)

        self.Gk = round(self.Gk / t)
        self.__change_name_Gk(self.Gk)

    def calculate_error(self):
        """Возвращает кортеж относительных погрешностей в %:
        1 - Жадный относительно максимума
        2 - Бережливый относительно максимума
        3 - Жадно-бережливый относительно максимума
        4 - Бережливо-жадный относительно максимума
        5 - T(k)G относительно максимума
        6 - G(k) относительно максимума
        7 - CTG относительно максимума"""

        opt_max = self[0].ans[-1]

        if opt_max == 0:
            return [0.0 for _ in range(self.NUM_ALGORITHMS_FOR_ERROR)]

        errors = [round(abs(opt_max - self[i + 2].ans[-1]) / opt_max * 100, 2)
                  for i in range(self.NUM_ALGORITHMS_FOR_ERROR)]

        return errors

    @staticmethod
    def __hungarian_max(P: np.matrix):
        return linear_sum_assignment(P, True)

    @staticmethod
    def __hungarian_min(P: np.matrix):
        return linear_sum_assignment(P, False)

    @staticmethod
    def __greedy_max(P: np.matrix):
        return Algorithms.__greedy(P, True)

    @staticmethod
    def __greedy_min(P: np.matrix):
        return Algorithms.__greedy(P, False)

    @staticmethod
    def __greedy(P: np.matrix, is_max: bool) -> tuple[list[int], list[int]]:
        n, m = P.shape
        used = [False] * m
        row_ind = [i for i in range(n)]
        col_ind = [0] * n

        for i in range(n):
            cur_ind = -1
            cur_val = -1.0 if is_max else float('inf')

            for j in range(m):
                if not used[j]:
                    if is_max:
                        if P[i, j] > cur_val:
                            cur_val = P[i, j]
                            cur_ind = j
                    else:
                        if P[i, j] < cur_val:
                            cur_val = P[i, j]
                            cur_ind = j
            col_ind[i] = cur_ind
            used[cur_ind] = True

        return row_ind, col_ind

    @staticmethod
    def __greedy_thrifty(P: np.matrix) -> tuple[list[int], list[int]]:
        n, m = P.shape
        theta = int(m / 2)
        used = [False] * m
        row_ind = [i for i in range(n)]
        col_ind = [0] * n
        for i in range(theta):
            cur_ind = -1
            cur_val = -1.0
            for j in range(m):
                if not used[j]:
                    if P[i, j] > cur_val:
                        cur_val = P[i, j]
                        cur_ind = j
            col_ind[i] = cur_ind
            used[cur_ind] = True
        for i in range(theta, n):
            cur_ind = -1
            cur_val = float('inf')

            for j in range(m):
                if not used[j]:
                    if P[i, j] < cur_val:
                        cur_val = P[i, j]
                        cur_ind = j
            col_ind[i] = cur_ind
            used[cur_ind] = True

        return row_ind, col_ind

    @staticmethod
    def __thrifty_greedy(P: np.matrix) -> tuple[list[int], list[int]]:
        n, m = P.shape
        theta = int(m / 2)
        used = [False] * m
        row_ind = [i for i in range(n)]
        col_ind = [0] * n
        for i in range(theta):
            cur_ind = -1
            cur_val = float('inf')

            for j in range(m):
                if not used[j]:
                    if P[i, j] < cur_val:
                        cur_val = P[i, j]
                        cur_ind = j
            col_ind[i] = cur_ind
            used[cur_ind] = True
        for i in range(theta, n):
            cur_ind = -1
            cur_val = -1.0

            for j in range(m):
                if not used[j]:
                    if P[i, j] > cur_val:
                        cur_val = P[i, j]
                        cur_ind = j
            col_ind[i] = cur_ind
            used[cur_ind] = True

        return row_ind, col_ind

    def __TkG(self, P: np.matrix) -> tuple[list[int], list[int]]:
        n, m = P.shape
        theta = n // 2
        optimal_k = -1
        max_sum = -1
        for k in range(0, n - theta + 1):
            used = [False] * m
            cur_sum = 0
            for i in range(theta):
                numbers = [(P[i, j], j) for j in range(m) if not used[j]]
                numbers.sort()
                cur_sum += numbers[k][0]
                used[numbers[k][1]] = True

            for i in range(theta, n):
                cur_ind = -1
                cur_val = -1.0
                for j in range(m):
                    if not used[j] and P[i, j] > cur_val:
                        cur_val = P[i, j]
                        cur_ind = j
                cur_sum += cur_val
                used[cur_ind] = True
            if cur_sum > max_sum:
                max_sum = cur_sum
                optimal_k = k

        used = [False] * m
        row_ind = [i for i in range(n)]
        col_ind = [0] * n

        for i in range(theta):
            numbers = [(P[i, j], j) for j in range(m) if not used[j]]
            numbers.sort()
            col_ind[i] = numbers[optimal_k][1]
            used[numbers[optimal_k][1]] = True

        for i in range(theta, n):
            cur_ind = -1
            cur_val = -1.0
            for j in range(m):
                if not used[j] and P[i, j] > cur_val:
                    cur_val = P[i, j]
                    cur_ind = j
            col_ind[i] = cur_ind
            used[cur_ind] = True

        self.TkG += optimal_k + 1
        return row_ind, col_ind

    def __Gk(self, P: np.matrix) -> tuple[list[int], list[int]]:
        n, m = P.shape
        optimal_k = -1
        max_sum = -1

        for k in range(1, n + 1):
            used = [False] * n
            cur_sum = 0

            for i in range(0, n, k):
                numbers = [(P[i, j], j) for j in range(m) if not used[j]]
                numbers.sort(reverse=True)
                for j in range(min(len(numbers), k)):
                    cur_sum += P[i + j, numbers[j][1]]
                    used[numbers[j][1]] = True

            if cur_sum > max_sum:
                max_sum = cur_sum
                optimal_k = k

        used = [False] * n
        row_ind = [i for i in range(n)]
        col_ind = [0] * n

        for i in range(0, n, optimal_k):
            numbers = [(P[i, j], j) for j in range(m) if not used[j]]
            numbers.sort(reverse=True)
            for j in range(min(len(numbers), optimal_k)):
                col_ind[i + j] = numbers[j][1]
                used[numbers[j][1]] = True

        self.Gk += optimal_k
        return row_ind, col_ind

    def __CTG(self, P: np.matrix) -> tuple[list[int], list[int]]:
        n, m = P.shape
        numbers = [(P[0, j], j) for j in range(m)]
        numbers.sort()
        row_ind = [i for i in range(n)]
        col_ind = [0] * n

        for i in range(n):
            if i < self.v - 1:
                col_ind[i] = numbers[n - 2 * self.v + 2 * (i + 1)][1]
            elif self.v - 1 <= i < 2 * self.v - 1:
                col_ind[i] = numbers[n + 2 * self.v - 2 * (i + 1) - 1][1]
            else:
                col_ind[i] = numbers[n - i - 1][1]
        return row_ind, col_ind
