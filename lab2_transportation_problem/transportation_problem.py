import numpy as np
import copy
from lab2_transportation_problem.cycle_subproblem import *


class TransportProblem:
    def __init__(self, A):
        self.rate_array = np.array(copy.deepcopy(A), dtype=int)
        self.s_b = np.sum(self.rate_array, axis=0)[0]
        self.s_a = np.sum(self.rate_array, axis=1)[0]
        self.n = len(A) - 1  # кол-во пунктов хранения
        self.m = len(A[0]) - 1  # кол-во пунктов назначения
        self.result_vec = []

        self.__create_supplies_array()

    # Создание таблицы решения
    def __create_supplies_array(self):
        self.supplies_array = np.array([[None for j in range(self.m + 1)] for i in range(self.n + 1)])

        for j in range(1, self.m + 1):
            self.supplies_array[0][j] = self.rate_array[0][j]

        for i in range(1, self.n + 1):
            self.supplies_array[i][0] = self.rate_array[i][0]

    # проверка на закрытость
    def is_closed_problem(self):
        return self.s_a == self.s_b

    # Преобразование задачи к закрытому типу
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

        for i in range(0, self.n):
            for j in range(0, self.m):

                if self.supplies_array[i][j] is None:
                    self.result_vec.append(0)
                else:
                    self.result_vec.append(self.supplies_array[i][j])

    # вычисление значения целевой функции
    def obj_function_value(self):
        s = 0

        for i in range(0, self.n):
            for j in range(0, self.m):
                res = self.supplies_array[i][j]
                if res is None:
                    res = 0

                s += self.rate_array[i + 1][j + 1] * res

        return s

    # проверка начального приближения на кол-во заполненых клеток(их должно быть m+n-1)
    def __is_initial_approximation_right(self):
        k = 0

        for elem in self.result_vec:
            if elem > 0:
                k = k + 1

        b = (self.m + self.n - 1) == k
        return b

    # метод северо-западного угла
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

        self.supplies_array = self.supplies_array[1:, 1:]

        self.__create_result_vector()

        if self.__is_initial_approximation_right() is False:
            print("Error!")

    # Вычисление потенциалов
    def __compute_potentials(self):
        self.v_potential = [None] * self.m
        self.u_potential = [None] * self.n

        for i in range(0, self.n):
            for j in range(0, self.m):
                if self.supplies_array[i][j] is not None:
                    if self.v_potential[j] is None and self.u_potential[i] is None:
                        self.u_potential[i] = 0
                        self.v_potential[j] = self.rate_array[i + 1][j + 1]
                    else:
                        if self.u_potential[i] is not None:
                            self.v_potential[j] = self.rate_array[i + 1][j + 1] + self.u_potential[i]
                        elif self.v_potential[j] is not None:
                            self.u_potential[i] = self.v_potential[j] - self.rate_array[i + 1][j + 1]

    # Проверка оптимальности на текущем шаге(Если v_j - u_i <= c_i,j, то решение оптимальное
    def __check_optimal_solution(self):
        max_abs = -1
        indexes_pair = None
        for i in range(0, self.n):
            for j in range(0, self.m):
                potentials_check = self.v_potential[j] - self.u_potential[i] - self.rate_array[i + 1, j + 1]

                if self.supplies_array[i, j] is None and potentials_check > 0:
                    if np.abs(potentials_check) > max_abs:
                        max_abs = np.abs(potentials_check)
                        indexes_pair = (i, j)

        return indexes_pair

    def __get_list_of_minuses(self, minuses_indexes):
        result_a = []
        for indexes in minuses_indexes:
            i, j = indexes
            result_a.append(self.supplies_array[i][j])

        return result_a

    def __recalculate_minuses(self, minus_indexes, minus_list, index_min_of_minuses, val_min_of_minuses):
        for k in range(0, len(minus_indexes)):
            i, j = minus_indexes[k]

            if k == index_min_of_minuses:
                self.supplies_array[i][j] = None  # Зануляем элемент матрицы, который был минимум из всех 'минусов'
            else:
                self.supplies_array[i][j] -= val_min_of_minuses  # А остальные 'минусы' просто уменьшаем на этот миниум

    def __recalculate_pluses(self, pluses_indexes, val_min_of_minuses):
        for k in range(0, len(pluses_indexes)):
            i, j = pluses_indexes[k]
            if k == 0:
                self.supplies_array[i, j] = val_min_of_minuses
            else:
                self.supplies_array[i, j] += val_min_of_minuses

    # Пересчет матрицы согласно 'минусам' и 'плюсам'
    def __recalculate_matrix(self, indexes_l):
        minus_indexes = [v for k, v in enumerate(indexes_l) if k % 2]
        plus_indexes = [v for k, v in enumerate(indexes_l) if not k % 2]

        minus_list = self.__get_list_of_minuses(minus_indexes)
        val_min_of_minuses, index_min_of_minuses = min((val, idx) for (idx, val) in enumerate(minus_list))

        self.__recalculate_minuses(minus_indexes, minus_list, index_min_of_minuses, val_min_of_minuses)
        self.__recalculate_pluses(plus_indexes, val_min_of_minuses)

    def potential_method(self):
        iteration_index = 0
        self.northwest_corner_method()

        while True:
            print('step ' + str(iteration_index))
            print(self.supplies_array)
            print('Objective function value: ' + str(self.obj_function_value()))

            self.__compute_potentials()
            indexes = self.__check_optimal_solution()

            if indexes is None:
                break

            print('___________________')
            i, j = indexes
            cs = CycleSubproblem(self.supplies_array, None)
            indexes_l = cs.find_cycle(i, j)

            if indexes_l is None:
                print("Error!")

            self.__recalculate_matrix(indexes_l)
            iteration_index += 1

        print('Done!')

    def __brute_force_method(self):
        print('brute_force')
