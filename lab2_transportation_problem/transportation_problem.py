import numpy as np
import copy


class TransportProblem:
    def __init__(self, A):
        self.rate_array = np.array(copy.deepcopy(A), dtype=int)
        self.s_b = np.sum(self.rate_array, axis=0)[0]
        self.s_a = np.sum(self.rate_array, axis=1)[0]
        self.n = len(A) - 1  # кол-во пунктов хранения
        self.m = len(A[0]) - 1  # кол-во пунктов назначения

        self.__create_supplies_array()

    def __create_supplies_array(self):
        self.supplies_array = [[0 for j in range(self.m + 1)] for i in range(self.n + 1)]

        for j in range(1, self.m + 1):
            self.supplies_array[0][j] = self.rate_array[0][j]

        for i in range(1, self.n + 1):
            self.supplies_array[i][0] = self.rate_array[i][0]


    def __is_closed_problem(self):
        return self.s_a == self.s_b

    def __to_closed_problem(self):
        if self.s_a > self.s_b:
            b_new = self.s_a - self.s_b
            self.s_b = self.s_a
            self.rate_array = np.r_[self.rate_array, [[b_new] + [0 for i in range(self.m)]]]

        elif self.s_a < self.s_b: # TODO: добавить штраф за недопоставку
            a_new = self.s_b - self.s_a
            self.s_a = self.s_b
            self.rate_array = np.c_[self.rate_array, [a_new] + [0 for i in range(self.n)]]

    def __northwest_corner_method(self):
        print('northwest')

    def __potential_method(self):
        print('potential')


    def __brute_force_method(self):
        print('brute_force')


