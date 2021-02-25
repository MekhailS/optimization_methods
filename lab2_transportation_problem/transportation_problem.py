import numpy as np
import copy


class TransportProblem:
    def __init__(self, A):
        self.rate_array = np.array(copy.deepcopy(A), dtype=int)
        self.s_b = np.sum(self.rate_array, axis=0)[0]
        self.s_a = np.sum(self.rate_array, axis=1)[0]
        self.n = len(A) - 1  # кол-во пунктов хранения
        self.m = len(A[0]) - 1  # кол-во пунктов назначения
        self.result_vec = []

        self.__create_supplies_array()

    def __create_supplies_array(self):
        # Воозможно стоит заполнять эту матрицу '-1', так как далее будет удобнее проверять заполненность поля
        self.supplies_array = np.array([[0 for j in range(self.m + 1)] for i in range(self.n + 1)])

        for j in range(1, self.m + 1):
            self.supplies_array[0][j] = self.rate_array[0][j]

        for i in range(1, self.n + 1):
            self.supplies_array[i][0] = self.rate_array[i][0]

    def is_closed_problem(self):
        return self.s_a == self.s_b

    def to_closed_problem(self):
        if self.s_a > self.s_b:
            b_new = self.s_a - self.s_b
            self.s_b = self.s_a
            self.rate_array = np.r_[self.rate_array, [[b_new] + [0 for i in range(self.m)]]]

        elif self.s_a < self.s_b:  # TODO: добавить штраф за недопоставку
            a_new = self.s_b - self.s_a
            self.s_a = self.s_b
            self.rate_array = np.c_[self.rate_array, [a_new] + [0 for i in range(self.n)]]

    # создание вектора решение из таблицы
    def __create_result_vector(self):
        self.result_vec.clear()

        for i in range(1, self.n + 1):
            for j in range(1, self.m + 1):
                self.result_vec.append(self.supplies_array[i][j])

    # вычисление значения целевой функции
    def obj_function_value(self):
        s = 0

        for i in range(1, self.n + 1):
            for j in range(1, self.m + 1):
                s += self.rate_array[i][j] * self.supplies_array[i][j]

        return s

    def __is_initial_approximation_right(self):
        k = 0

        for elem in self.result_vec:
            if elem > 0:
                k = k + 1

        b = (self.m + self.n - 1) == k
        return b

    # TODO: нужны ли нам далее a_i и b_j? В этом методе они зануляются
    #       Возможно мы туда просто вставим потенциалы
    def northwest_corner_method(self):

        def __cell_value(i, j):
            min_el = np.minimum(self.supplies_array[i][0], self.supplies_array[0][j])
            self.supplies_array[0][j] -= min_el
            self.supplies_array[i][0] -= min_el
            self.supplies_array[i][j] = min_el

        j = 1

        for k in range(1, self.n + 1):
            while self.supplies_array[k][0] != 0:
                __cell_value(k, j)
                j += 1
            j -= 1

        self.__create_result_vector()

        if self.__is_initial_approximation_right() is False:
            print("Error!")

    def potential_method(self):
        self.northwest_corner_method()
        print(self.supplies_array)

        v_potential = [None] * self.m
        u_potential = [None] * self.n


        for i in range(0, self.n):
            for j in range(0, self.m):
                if self.supplies_array[i + 1][j + 1] != 0:
                    if v_potential[j] is None and u_potential[i] is None:
                        u_potential[i] = 0
                        v_potential[j] = self.rate_array[i + 1][j + 1]
                    else:
                        if u_potential[i] is not None:
                            v_potential[j] = self.rate_array[i + 1][j + 1] + u_potential[i]
                        elif v_potential[j] is not None:
                            u_potential[i] = v_potential[j] - self.rate_array[i + 1][j + 1]

        print('Potnetial')

    def __brute_force_method(self):
        print('brute_force')
