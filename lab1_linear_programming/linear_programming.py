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

        tableau = self.__make_tableau()

        canonical_tableau = None

        if not self.__is_canonical_tableau(tableau):
            tableau_with_artificial_vars = self.__make_tableau_with_artificial_vars(tableau)
            canonical_tableau = self.__extract_canonical_tableau(tableau_with_artificial_vars)
            if canonical_tableau is None:
                return None, None

        else:
            canonical_tableau = tableau

        self.__process_tableau(canonical_tableau)
        my_res = self.__extract_solutions(canonical_tableau)
        res_scipi = linprog(method='simplex', c=self.c_objective, A_eq=self.A, b_eq=self.b)

    def __make_tableau(self):
        new_A = self.A.copy()
        new_b = self.b.copy()

        for i in range(new_b.shape[0]):
            if new_b[i] < 0:
                new_b[i] = -new_b[i]
                new_A[i, :] = -new_A[i, :]

        tableau = np.insert(new_A, 0, -self.c_objective, axis=0)
        first_column = np.zeros(tableau.shape[0])
        first_column[0] = 1.0
        tableau = np.insert(tableau, 0, first_column, axis=1)
        last_column = np.insert(new_b, 0, 0.0)
        tableau = np.insert(tableau, tableau.shape[1], last_column, axis=1)
        return tableau

    def __is_canonical_tableau(self, tableau):
        basic_variables = self.__find_basic_variables(tableau)

        num_of_equations = tableau.shape[0] - 1

        return len(basic_variables) == len(num_of_equations)

    def __make_tableau_with_artificial_vars(self, tableau):
        canonical_tableau = tableau.copy()

        basic_variables = self.__find_basic_variables(canonical_tableau)[0]

        num_of_equations = canonical_tableau.shape[0] - 1

        eye_matrix = np.eye(num_of_equations)

        matrix_of_existing_basis = np.zeros((num_of_equations, num_of_equations))
        k = num_of_equations - 1
        for i in basic_variables:
            matrix_of_existing_basis[:, k] = canonical_tableau[1:, i]
            k -= 1

        matrix_of_artificial_variables = eye_matrix - matrix_of_existing_basis
        idexes_of_zero_columns = np.argwhere(np.all(matrix_of_artificial_variables[..., :] == 0, axis=0))
        matrix_of_artificial_variables = np.delete(matrix_of_artificial_variables, idexes_of_zero_columns, axis=1)

        zero_row = [0] * matrix_of_artificial_variables.shape[1]
        matrix_of_artificial_variables = np.insert(matrix_of_artificial_variables, 0, zero_row, axis=0)

        for i in range(matrix_of_artificial_variables.shape[1]):
            canonical_tableau = np.insert(canonical_tableau, canonical_tableau.shape[1] - 1,
                                          matrix_of_artificial_variables[:, i], axis=1)

        first_row = np.zeros(canonical_tableau.shape[1])
        for i in range(2, len(matrix_of_artificial_variables)):
            first_row[-i] = -1
        canonical_tableau = np.insert(canonical_tableau, 0, first_row, axis=0)

        first_column = np.zeros(canonical_tableau.shape[0])
        first_column[0] = 1.0
        canonical_tableau = np.insert(canonical_tableau, 0, first_column, axis=1)

        return canonical_tableau

    def __extract_canonical_tableau(self, canonical_tableau):
        first_row = canonical_tableau[0, :]

        row_indices = []

        # to remember where was the artificial variables
        artificial_col_indices = []

        for col_num in range(len(first_row)):
            if first_row[col_num] == -1:
                marker = has_only_one_unit_and_zeros(canonical_tableau[1:, col_num])
                if marker.get('flag'):
                    row_indices.append(marker.get('row_idx') + 1)

                    artificial_col_indices.append(col_num)

        for row_idx in row_indices:
            canonical_tableau[0, :] = canonical_tableau[0, :] + canonical_tableau[row_idx, :]

        self.__process_tableau(canonical_tableau)
        res = self.__extract_solutions(canonical_tableau)

        # not sure
        for i in artificial_col_indices:
            if res[1][i] > 0:
                return None

        # if all good return canonical_tableau
        canonical_tableau = np.delete(canonical_tableau, artificial_col_indices, axis=1)
        canonical_tableau = np.delete(canonical_tableau, 0, axis=1)
        canonical_tableau = np.delete(canonical_tableau, 0, axis=0)

        return canonical_tableau

    def __process_tableau(self, tableau):
        idx_of_pivot_row, idx_of_pivot_column = self.__select_pivot_row_and_column(tableau)
        while idx_of_pivot_row is not None and idx_of_pivot_column is not None:
            self.__change_basic_variables(tableau, idx_of_pivot_row, idx_of_pivot_column)

            idx_of_pivot_row, idx_of_pivot_column = self.__select_pivot_row_and_column(tableau)

    # def __find_basic_variables(self, tableau):
    #     basic_variables = []
    #     for col in range(1, tableau.shape[1] - 1):
    #         if has_only_one_unit_and_zeros(tableau[:, col].get('flag')):
    #             basic_variables.append(col)
    #     return list(basic_variables)
    #
    # def __find_non_basic_variables(self, tableau):
    #     return list(set(range(1, tableau.shape[1] - 1)) - set(self.__find_basic_variables(tableau)))

    def __find_basic_variables(self, tableau):
        # columns which have only one non-zero element
        one_nonzero_col = []
        for col in range(1, tableau.shape[1] - 1):
            num_of_elements = (tableau[:, col] != 0).sum()
            nonzero_idx = list(tableau[:, col].nonzero()[0])
            shit = tableau[nonzero_idx[0], -1]
            if len(nonzero_idx) == 1 and tableau[nonzero_idx[0], col] * tableau[
                nonzero_idx[0], -1] >= 0:
                one_nonzero_col.append(col)

        A_one_nonzero = tableau[:, one_nonzero_col]
        # indexes of non-zero elements in one_nonzero_columns
        idx_val_nonzero = list((A_one_nonzero != 0).argmax(axis=0))
        # values of non-zero elements
        nonzero_values = np.array(
            [tableau[idx_val_nonzero[i], one_nonzero_col[i]] for i in range(len(one_nonzero_col))])

        # construct following table 'nonzero_table':
        # column index of nonzero element:        ---one_nonzero_col----
        # row index of nonzero element in column: ---idx_val_nonzero---
        # value of nonzero elements:              ---nonzero_values----
        nonzero_table = np.array([list(one_nonzero_col),
                                  list(idx_val_nonzero),
                                  list(nonzero_values)])

        ''' # get unique values of nonzero elements and their frequency
        unique_val, freq_count = np.unique(nonzero_values, return_counts=True)
        dict_frequency_count = dict(zip(list(unique_val),
                                        list(freq_count)))

        # we only interested in unique values which have freq >= number of equations in lp problem
        frequency_all_elements = np.array([dict_frequency_count[val] for val in list(nonzero_values)])
        nonzero_table = nonzero_table[:, frequency_all_elements >= self.tableau.shape[0]-1]
        one_nonzero_col = nonzero_table[0, :]
        idx_val_nonzero = nonzero_table[1, :]
        nonzero_values = nonzero_table[2, :]

        # this way, from one_nonzero_col we deleted columns which could not form
        # matrix of form 'scalar*E', because number of these deleted columns
        # wasn't enough to form matrix 'scalar*E' '''

        # now check for each nonzero value if we can form matrix nonzero_value*E
        # (if there are all necessary idx_val_nonzero)

        unique_values = np.unique(nonzero_values)

        best_nonzero_table = np.array([[]])
        max_num_nonzero_col = 0
        best_nonzero_val = None
        for val in unique_values:
            val_nonzero_table = nonzero_table[:, nonzero_values == val]
            idx_nonzero_val = set(val_nonzero_table[1, :])

            if len(idx_nonzero_val) > max_num_nonzero_col:
                max_num_nonzero_col = len(idx_nonzero_val)
                best_nonzero_table = val_nonzero_table
                best_nonzero_val = val

        if best_nonzero_val is None:
            basic_variables = None
            non_basic_variables = list(range(1, tableau.shape[1] - 1))
            return None

        base_table = best_nonzero_table[best_nonzero_table[1, :].argsort()][:, :max_num_nonzero_col]
        basic_variables = list(base_table[0, :].astype(int))
        non_basic_variables = list(set(range(1, tableau.shape[1] - 1)) - set(basic_variables))

        return basic_variables, base_table[1, :], best_nonzero_val

    def __find_non_basic_variables(self, tableau):
        return list(set(range(1, tableau.shape[1] - 1)) - set(self.__find_basic_variables(tableau)[0]))

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


def has_only_one_unit_and_zeros(array):
    d = dict()
    num_of_units = 1
    row_idx = -1
    for elem in array:
        if elem != 1 and elem != 0:
            num_of_units += 1
        if elem == 1:
            row_idx = elem.__index__

    if num_of_units == 1:
        d['flag'] = True
        d['row_idx'] = row_idx
    else:
        d['flag'] = False
        d['row_idx'] = None

    return d


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
