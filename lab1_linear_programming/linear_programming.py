from enum import Enum
import numpy as np
import copy


class LPProblem:
    def __init__(self, x_dim, A, b, c_objective,
                 M1_b_ineq=None, N1_x_positive=None):

        if M1_b_ineq is None:
            M1_b_ineq = []
        if N1_x_positive is None:
            N1_x_positive = []

        self.x_dim = x_dim
        self.A = np.array(A)
        self.b = np.array(b)
        self.c_objective = np.array(c_objective)

        self.M1_b_ineq = list(M1_b_ineq)
        self.M2_b_eq = list(set(range(self.b.shape[0])) - set(M1_b_ineq))
        self.N1_x_positive = list(N1_x_positive)
        self.N2_x_any_sign = list(set(range(x_dim)) - set(N1_x_positive))

    def from_common_to_dual(self, inplace=False):
        dual = copy.deepcopy(self)

        dual.b, dual.c_objective = dual.c_objective, dual.b
        dual.A = np.transpose(dual.A)
        dual.M1_b_ineq, dual.M2_b_eq, dual.N1_x_positive, dual.N2_x_any_sign = dual.N1_x_positive, dual.N2_x_any_sign, dual.M1_b_ineq, dual.M2_b_eq

        if inplace:
            self.__dict__.update(dual.__dict__)

        return dual

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

    def is_canonical(self):
        return len(self.M1_b_ineq) == 0 and self.x_dim == len(self.N1_x_positive)

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

        if inplace:
            self.__dict__.update(res.__dict__)

        return res
