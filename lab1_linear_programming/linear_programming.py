import numpy as np
import copy
from itertools import combinations
from enum import Enum
from lab1_linear_programming.simplex_algorithm import *
from scipy.optimize import linprog


def is_full_rank(A):
    m = len(A)
    if m == 0:
        return True

    rank_matrix = np.linalg.matrix_rank(A)
    return rank_matrix == m


class ObjectiveDirection(Enum):
    MIN = 1
    MAX = -1


class LPProblem:
    def __init__(self, x_dim, A, b, c_objective,
                 M1_b_ineq=None, N1_x_positive=None,
                 obj_direction=ObjectiveDirection.MIN):

        if M1_b_ineq is None:
            M1_b_ineq = []
        if N1_x_positive is None:
            N1_x_positive = []

        self.obj_direction = obj_direction
        self.x_dim = x_dim
        self.A = np.array(copy.deepcopy(A), dtype=float)
        self.b = np.array(copy.deepcopy(b), dtype=float)
        self.c_objective = np.array(copy.deepcopy(c_objective), dtype=float)
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

    def __update_rank(self):
        self.full_rank = is_full_rank(self.A)
        return self.full_rank

    def is_canonical(self):
        return len(self.M1_b_ineq) == 0 and \
               self.x_dim == len(self.N1_x_positive) and \
               (self.full_rank if self.full_rank is not None else self.__update_rank())

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
            N1_x_positive=list(range(x_new_dim)),
            obj_direction=self.obj_direction
        )
        if self.full_rank is True or is_full_rank(canon.A[self.M2_b_eq, :]):
            res.full_rank = True

        if inplace:
            self.__dict__.update(res.__dict__)

        return res

    def dual(self, inplace=False):
        dual_A = -np.transpose(self.A)
        dual_b, dual_c = -self.c_objective, -self.b
        dual_x_dim = len(dual_c)
        dual_M1_b_ineq, dual_N1_x_positive = self.N1_x_positive, self.M1_b_ineq

        dual_problem = LPProblem(
            x_dim=dual_x_dim,
            A=dual_A,
            b=dual_b,
            c_objective=dual_c,
            M1_b_ineq=dual_M1_b_ineq,
            N1_x_positive=dual_N1_x_positive,
            obj_direction=ObjectiveDirection.MIN
        )

        if inplace:
            self.__dict__.update(dual_problem.__dict__)

        return dual_problem

    def __solve_canon_extreme_points_bruteforce(self):
        if not self.is_canonical():
            ValueError('LP problem is not canonical')
            return None, []

        solutions_potential = []
        for comb in combinations(list(self.N1_x_positive), self.A.shape[0]):
            x_ls = solve_linear_system_gauss(self.A[:, list(comb)], self.b)
            if x_ls is None or not (np.isfinite(x_ls)).all() or (x_ls < 0).any():
                continue

            x = np.zeros(self.x_dim)
            x[list(comb)] = x_ls
            solutions_potential.append((x, self.c_objective @ x))

        if len(solutions_potential) == 0:
            return None, []

        solutions_potential.sort(key=lambda elem: elem[1])
        solutions_potential = [el[0] for el in solutions_potential]

        return solutions_potential[0], np.array(solutions_potential)

    class SolvingMethod(str, Enum):
        SIMPLEX = 'simplex'
        BRUTEFORCE = 'bruteforce'
        SCIPY = 'scipy'

    def solve(self, mode=SolvingMethod.BRUTEFORCE):
        lp_canonical = self.canonical(inplace=False)
        x, x_path = None, []

        if mode == self.SolvingMethod.BRUTEFORCE:
            x, x_path = lp_canonical.__solve_canon_extreme_points_bruteforce()
        elif mode == self.SolvingMethod.SIMPLEX:
            simplex_alg = SimplexAlgorithm(
                A=lp_canonical.A,
                b=lp_canonical.b,
                c=lp_canonical.c_objective
            )
            x, x_path = simplex_alg.solve()
            if mode == self.SolvingMethod.SIMPLEX:
                x, x_path = lp_canonical.__solve_canon_extreme_points_bruteforce()

                pass
        elif mode == self.SolvingMethod.SCIPY:
            res = linprog(A_eq=lp_canonical.A, b_eq=lp_canonical.b,
                          c=lp_canonical.c_objective)
            x = res.x

        def transform_canonical_solution(x):
            if x is None:
                return None

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
        if A[i_piv, i_piv] == 0:
            return None
        x[i_piv] = (b[i_piv] - np.dot(A[i_piv, i_piv + 1:], x[i_piv + 1:])) / A[i_piv, i_piv]
    return x
