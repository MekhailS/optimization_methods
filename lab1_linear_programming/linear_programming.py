import numpy as np
import copy
from itertools import combinations
from enum import Enum
import sys
from scipy.optimize import linprog


def is_full_rank(A):
    m = len(A)

    if m == 0:
        return True
    
    rank_matrix = np.linalg.matrix_rank(A)
    return rank_matrix == m


class LPProblem:
    def __init__(self, x_dim, A, b, c_objective,
                 M1_b_ineq=None, N1_x_positive=None):

        if M1_b_ineq is None:
            M1_b_ineq = []
        if N1_x_positive is None:
            N1_x_positive = []

        self.x_dim = x_dim
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c_objective = np.array(c_objective, dtype=float)
        self.full_rank = None

        self.M1_b_ineq = list(M1_b_ineq)
        self.M2_b_eq = list(set(range(self.b.shape[0])) - set(M1_b_ineq))
        self.N1_x_positive = list(N1_x_positive)
        self.N2_x_any_sign = list(set(range(x_dim)) - set(N1_x_positive))

        self.dual_problem = self.__from_common_to_dual()

    def __A_b_equality_part(self):
        if len(self.M2_b_eq) == 0:
            return None, None

        return self.A[self.M2_b_eq, :], self.b[self.M2_b_eq]

    def __A_b_inequality_part(self):
        if len(self.M1_b_ineq) == 0:
            return None, None

        return self.A[self.M1_b_ineq, :], self.b[self.M1_b_ineq]

    def optimization_area_indicator(self):
        # maybe inner function is redundant (refactor?)
        A_eq, b_eq = self.__A_b_equality_part()
        A_ineq, b_ineq = self.__A_b_inequality_part()

        def area_indicator(x, eps):
            x = np.array(x)

            is_solution_to_eq = True
            if A_eq is not None:
                eq_Ax = A_eq @ x
                distance_eq_Ax_b = np.linalg.norm(eq_Ax - b_eq)
                is_solution_to_eq = distance_eq_Ax_b <= eps

            is_solution_to_ineq = True
            if A_ineq is not None:
                ineq_Ax = A_ineq @ x
                distance_ineq_Ax_b = np.linalg.norm(ineq_Ax - b_ineq)
                is_solution_to_ineq = (distance_ineq_Ax_b <= eps or (ineq_Ax >= b_ineq).all())

            return (len(self.N1_x_positive) == 0 or (x[self.N1_x_positive] >= 0).all()) and \
                   is_solution_to_eq and is_solution_to_ineq

        return area_indicator

    def objective_func(self):
        # maybe inner function is redundant (refactor?)
        def obj_func(x):
            return self.c_objective @ np.array(x)

        return obj_func

    def update_rank(self):
        self.full_rank = is_full_rank(self.A)
        return self.full_rank

    def is_canonical(self):
        return len(self.M1_b_ineq) == 0 and \
               self.x_dim == len(self.N1_x_positive) and \
               (self.full_rank if self.full_rank is not None else self.update_rank())

    def canonical(self, inplace=False):
        canon = copy.deepcopy(self)

        if canon.is_canonical():
            return canon

        A_m1_n1 = canon.A[self.M1_b_ineq][:, self.N1_x_positive]
        A_m1_n2 = canon.A[self.M1_b_ineq][:, self.N2_x_any_sign]
        neg_A_m1_n2 = A_m1_n2.copy() * -1
        neg_E_m1_m1 = np.eye(len(self.M1_b_ineq)) * -1

        A_m2_n1 = canon.A[self.M2_b_eq][:, self.N1_x_positive]
        A_m2_n2 = canon.A[self.M2_b_eq][:, self.N2_x_any_sign]
        neg_A_m2_n2 = A_m2_n2.copy() * -1
        O_m2_m1 = np.zeros([len(self.M2_b_eq), len(self.M1_b_ineq)])

        A_upper = np.concatenate((A_m1_n1, A_m1_n2, neg_A_m1_n2, neg_E_m1_m1), axis=1)
        A_lower = np.concatenate((A_m2_n1, A_m2_n2, neg_A_m2_n2, O_m2_m1), axis=1)

        A = np.concatenate((A_upper, A_lower), axis=0)

        b = canon.b

        c_n1 = canon.c_objective[self.N1_x_positive]
        c_n2 = canon.c_objective[self.N2_x_any_sign]
        neg_c_n2 = c_n2.copy() * -1
        o_m1 = np.zeros(len(self.M1_b_ineq))

        c = np.concatenate((c_n1, c_n2, neg_c_n2, o_m1), axis=0)

        x_new_dim = A.shape[1]
        res = LPProblem(
            x_dim=x_new_dim,
            A=A,
            b=b,
            c_objective=c,
            M1_b_ineq=None,
            N1_x_positive=list(range(x_new_dim))
        )
        if self.full_rank is True or is_full_rank(canon.A[self.M2_b_eq, :]):
            res.full_rank = True

        if inplace:
            self.__dict__.update(res.__dict__)

        return res

    def __from_common_to_dual(self, inplace=False):
        dual = copy.deepcopy(self)

        dual.b, dual.c_objective = dual.c_objective, dual.b
        dual.x_dim = len(dual.c_objective)

        dual.A = np.transpose(dual.A)

        dual.M1_b_ineq, dual.M2_b_eq, dual.N1_x_positive, dual.N2_x_any_sign = dual.N1_x_positive, dual.N2_x_any_sign, dual.M1_b_ineq, dual.M2_b_eq

        if inplace:
            self.__dict__.update(dual.__dict__)

        return dual

    def __solve_canon_extreme_points_bruteforce(self):
        if not self.is_canonical():
            ValueError('LP problem is not canonical')
            return None, None

        solutions_potential = []
        for comb in combinations(list(self.N1_x_positive), self.A.shape[0]):
            x_ls = solve_linear_system_gauss(self.A[:, list(comb)], self.b)
            if x_ls is None or (x_ls < 0).any():
                continue

            x = np.zeros(self.x_dim)
            x[list(comb)] = x_ls
            solutions_potential.append((x, self.c_objective @ x))

        if len(solutions_potential) == 0:
            return None, None

        solutions_potential.sort(key=lambda elem: elem[1])
        solutions_potential = [el[0] for el in solutions_potential]

        return solutions_potential[0], np.array(solutions_potential)

    def __solve_canon_simplex(self):
        if not self.is_canonical():
            ValueError('LP problem is not canonical')
            return None, None
        canonical_tableau = self.__make_tableau()


        idx_of_pivot_row, idx_of_pivot_column = self.__select_pivot_row_and_column(canonical_tableau)
        while idx_of_pivot_row is not None and idx_of_pivot_column is not None:
            self.__change_basic_variables(canonical_tableau, idx_of_pivot_row, idx_of_pivot_column)

            idx_of_pivot_row, idx_of_pivot_column = self.__select_pivot_row_and_column(canonical_tableau)

        res_scipi = linprog(method='simplex', c=self.c_objective, A_eq=self.A, b_eq=self.b)
        misha_res, stuff = self.__solve_canon_extreme_points_bruteforce()
        my_res = self.__extract_solutions(canonical_tableau)
        print(f'{my_res[1] - res_scipi.x}')


    def __make_tableau(self):

        new_A = self.A.copy()
        new_b = self.b.copy()

        # for i in range(new_b.shape[0]):
        #     if new_b[i] < 0:
        #         new_b[i] = -new_b[i]
        #         new_A[i, :] = -new_A[i, :]

        canonical_tableau = np.insert(new_A, 0, -self.c_objective, axis=0)
        first_column = np.zeros(canonical_tableau.shape[0])
        first_column[0] = 1.0
        canonical_tableau = np.insert(canonical_tableau, 0, first_column, axis=1)
        last_column = np.insert(new_b, 0, 0.0)
        canonical_tableau = np.insert(canonical_tableau, canonical_tableau.shape[1], last_column, axis=1)
        return canonical_tableau

    # def __make_canonical_tableau(self, tableau):
    #     basic_variables = self.__find_basic_variables(tableau)
    #
    #     num_of_equations = tableau.shape[0] - 1
    #
    #
    #     if len(basic_variables) == 0:
    #         eye_matrix = np.eye(num_of_equations)
    #         tableau.insert(eye_matrix, axis=0)
    #         artificial_objective_function =
    #
    #     return None

    def __find_basic_variables(self, tableau):
        basic_variables = []
        for col in range(1, tableau.shape[1] - 1):
            num_of_elements = (np.abs(tableau[:, col]) != 0).sum()
            if num_of_elements == 1:
                basic_variables.append(col)
        return list(basic_variables)
        # basic_variables = []
        # for col in range(1, tableau.shape[1] - 1):
        #     num_of_units = 1
        #     for elem in tableau[:, col]:
        #         if elem != 1 and elem != 0:
        #             num_of_units += 1
        #     if num_of_units == 1:
        #         basic_variables.append(col)
        # return list(basic_variables)

    def __find_non_basic_variables(self, tableau):
        return list(set(range(1, tableau.shape[1] - 1)) - set(self.__find_basic_variables(tableau)))

    def __select_pivot_row_and_column(self, tableau):
        first_row = tableau[0, :]
        non_basic_variables = self.__find_non_basic_variables(tableau)

        if all(first_row[i] <= 0 for i in non_basic_variables):
            return None, None

        idx_of_pivot_column = find_max(first_row[1:], non_basic_variables) + 1

        b = tableau[1:, tableau.shape[1] - 1]
        a = tableau[1:, idx_of_pivot_column]
        arr = b / a

        min_idx = find_min_non_negative(arr)
        if min_idx == None:
            return None, None

        idx_of_pivot_row = min_idx + 1

        return idx_of_pivot_row, idx_of_pivot_column

    def __change_basic_variables(self, tableau, idx_of_pivot_row, idx_of_pivot_column):
        pivot_value = tableau[idx_of_pivot_row][idx_of_pivot_column]

        for row_idx in range(tableau.shape[0]):
            if row_idx == idx_of_pivot_row:
                continue
            factor = tableau[row_idx][idx_of_pivot_column] / pivot_value
            tableau[row_idx] = tableau[row_idx] - tableau[idx_of_pivot_row] * factor

    def __extract_solutions(self, tableau):
        basic_variables = self.__find_basic_variables(tableau)
        non_basic_variables = self.__find_non_basic_variables(tableau)

        solutions = [0.0] * (len(basic_variables) + len(non_basic_variables))

        last_column = tableau.shape[1] - 1

        for i in basic_variables:
            row_idx = np.nonzero(tableau[:, i])[0][0]
            value_b = tableau[row_idx][last_column]
            value_a = tableau[row_idx][i]
            solution_value = value_b / value_a

            solutions[i - 1] = 0 if solution_value < 0 else solution_value

        minimum_value = tableau[0][last_column] / tableau[0][0]
        return minimum_value, solutions

    class SolvingMethod(str, Enum):
        SIMPLEX = 'simplex'
        BRUTEFORCE = 'bruteforce'

    # TODO: currently works not in really correct way
    #       (if 'self' is canonical problem)
    def solve(self, mode=SolvingMethod.BRUTEFORCE):
        lp_canonical = self.canonical(inplace=False)
        x, x_path = None, None

        if mode == self.SolvingMethod.BRUTEFORCE:
            x, x_path = lp_canonical.__solve_canon_extreme_points_bruteforce()
        elif mode == self.SolvingMethod.SIMPLEX:
            x, x_path = lp_canonical.__solve_canon_simplex()

        def transform_canonical_solution(x):
            u_and_v = x[len(self.N1_x_positive):-len(self.M1_b_ineq)]
            u = u_and_v[:len(self.N2_x_any_sign)]
            v = u_and_v[len(self.N2_x_any_sign):]
            x_sol = np.zeros(self.x_dim)
            x_sol[self.N1_x_positive] = x[:len(self.N1_x_positive)]
            x_sol[self.N2_x_any_sign] = u - v
            return x_sol

        return transform_canonical_solution(x), \
               [transform_canonical_solution(el) for el in x_path]


def find_max(array, indexes):
    max_value = 1
    first_positive_elem_idx = next(x for x, val in enumerate(array) if val > 0)
    max_index = first_positive_elem_idx
    for i in indexes:
        if array[i] > max_value:
            max_value = array[i]
            max_index = i
    return max_index

def find_min_non_negative(arr):
    min_value = sys.float_info.max
    min_idx = 0
    for i in range(len(arr)):
        if 0 < arr[i] < min_value:
            min_value = arr[i]
            min_idx = i
    return None if min_value == sys.float_info.max else min_idx


def same_sign(x, y):
    return True if x * y >= 0 else False


def solve_linear_system_gauss(A, b):
    A = np.array(A).copy()
    b = np.array(b).copy()

    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError('Invalid sizes of A and b')

    for i_piv in range(n - 1):
        max_index = abs(A[i_piv:, i_piv]).argmax() + i_piv
        if A[max_index, i_piv] == 0:
            return None

        if max_index != i_piv:
            A[[i_piv, max_index], :] = A[[max_index, i_piv], :]
            b[[i_piv, max_index]] = b[[max_index, i_piv]]

        for row in range(i_piv + 1, n):
            multiplier = A[row][i_piv] / A[i_piv][i_piv]

            A[row, i_piv:] = A[row, i_piv:] - multiplier * A[i_piv, i_piv:]
            b[row] = b[row] - multiplier * b[i_piv]

    x = np.zeros(n)
    for i_piv in range(n - 1, -1, -1):
        x[i_piv] = (b[i_piv] - np.dot(A[i_piv, i_piv + 1:], x[i_piv + 1:])) / A[i_piv, i_piv]
    return x
