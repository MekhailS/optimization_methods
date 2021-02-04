import numpy as np
import copy
from itertools import combinations
from enum import Enum


# TODO: check if all rows of given matrix A are linearly independent
#       e.g. rank(A[M, N]) == m (A has full rank)
def is_full_rank(A):
    return True


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
        return len(self.M1_b_ineq) == 0 and\
               self.x_dim == len(self.N1_x_positive) and\
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

    # TODO: dual problem must be constructed though constructor(!) of LPProplem
    #       (currently inside-features of common problem are copying to dual
    #       x_dim, full_rank)
    def from_common_to_dual(self, inplace=False):
        dual = copy.deepcopy(self)

        dual.b, dual.c_objective = dual.c_objective, dual.b
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

    # TODO : simplex algorithm
    def __solve_canon_simplex(self):
        if not self.is_canonical():
            ValueError('LP problem is not canonical')
            return None, None

        return None, None

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


def solve_linear_system_gauss(A, b):
    A = np.array(A).copy()
    b = np.array(b).copy()

    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError('Invalid sizes of A and b')

    for i_piv in range(n-1):
        max_index = abs(A[i_piv:, i_piv]).argmax() + i_piv
        if A[max_index, i_piv] == 0:
            return None

        if max_index != i_piv:
            A[[i_piv, max_index], :] = A[[max_index, i_piv], :]
            b[[i_piv, max_index]] = b[[max_index, i_piv]]

        for row in range(i_piv+1, n):
            multiplier = A[row][i_piv]/A[i_piv][i_piv]

            A[row, i_piv:] = A[row, i_piv:] - multiplier*A[i_piv, i_piv:]
            b[row] = b[row] - multiplier*b[i_piv]

    x = np.zeros(n)
    for i_piv in range(n - 1, -1, -1):
        x[i_piv] = (b[i_piv] - np.dot(A[i_piv, i_piv + 1:], x[i_piv + 1:])) / A[i_piv, i_piv]
    return x
