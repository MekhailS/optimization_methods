from enum import Enum
import numpy as np
import copy


class LinearSystem:

    class Relationship(str, Enum):
        equality = '='
        inequality_greater = '>='
        inequality_less = '<='

    def __init__(self, A=None, b=None, relationship=Relationship.equality,
                 transform_inequality_less=True):
        if A is None:
            A = []
        if b is None:
            b = []

        self.A = np.array(A)
        self.b = np.array(b)
        self.relationship = relationship
        if self.relationship == self.Relationship.inequality_less and transform_inequality_less:
            self.A *= -1
            self.b *= -1
            self.relationship = self.Relationship.inequality_greater

        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError('shapes of A and b dont match')

    def col_num(self):
        return self.A.shape[1]

    def row_num(self):
        return self.A.shape[0]

    def select_A_columns(self, col_list):
        return self.A[:, col_list].copy()

    def is_solution(self, x, eps):
        Ax = self.A @ x
        distance_Ax_b = np.linalg.norm(self.A @ x - self.b)

        if self.relationship == self.Relationship.equality:
            return distance_Ax_b <= eps

        elif self.relationship == self.Relationship.inequality_greater:
            return distance_Ax_b <= eps or (Ax > self.b).all()

        elif self.relationship == self.Relationship.inequality_less:
            return distance_Ax_b <= eps or (Ax < self.b).all()

    def foreach_A(self, func, inplace=True):
        vec_func = np.vectorize(func)
        res = vec_func(self.A.copy())

        if inplace:
            self.A = res

        return LinearSystem(res, self.b.copy())

    def foreach_b(self, func, inplace=True):
        vec_func = np.vectorize(func)
        res = vec_func(self.b.copy())

        if inplace:
            self.b = res

        return LinearSystem(self.A.copy(), res)


class LPProblem:

    def __init__(self, x_size=0,
                 LS_eq=None, LS_ineq=None, x_positive_indexes=None, c_objective=None):

        if x_positive_indexes is None:
            x_positive_indexes = []

        if (LS_eq is not None and LS_eq.col_num() != x_size) or \
                (LS_ineq is not None and LS_ineq.col_num() != x_size) or \
                (c_objective is None or len(c_objective) != x_size):
            raise ValueError('shape of x vector doesnt match system or objective vector')

        self.LS_eq = LS_eq
        self.LS_ineq = LS_ineq
        self.x_positive_indexes = sorted(x_positive_indexes)
        self.x_size = x_size
        self.c_objective = np.array(c_objective)

    def optimization_area_indicator(self):
        # maybe inner function is redundant (refactor?)
        def area_indicator(x, eps):
            x = np.array(x)
            return (len(self.x_positive_indexes) == 0 or (x[self.x_positive_indexes] >= 0).all()) and \
                   (self.LS_eq is None or self.LS_eq.is_solution(x, eps)) and \
                   (self.LS_ineq is None or self.LS_ineq.is_solution(x, eps))

        return area_indicator

    def objective_func(self):
        # maybe inner function is redundant (refactor?)
        def obj_func(x):
            return self.c_objective @ np.array(x)

        return obj_func

    def is_canonical(self):
        return self.LS_ineq is None and self.x_size == len(self.x_positive_indexes)

    def canonical(self, inplace=False):
        if self.is_canonical():
            return copy.deepcopy(self)

        x_any_sign_indexes = list(set(range(self.x_size)) - set(self.x_positive_indexes))

        A_m1_n1 = self.LS_ineq.select_A_columns(self.x_positive_indexes)
        A_m1_n2 = self.LS_ineq.select_A_columns(x_any_sign_indexes)
        neg_A_m1_n2 = A_m1_n2.copy() * -1
        neg_E_m1_m1 = np.eye(self.LS_ineq.row_num()) * -1

        A_m2_n1 = self.LS_eq.select_A_columns(self.x_positive_indexes)
        A_m2_n2 = self.LS_eq.select_A_columns(x_any_sign_indexes)
        neg_A_m2_n2 = A_m2_n2.copy() * -1
        O_m2_m1 = np.zeros([self.LS_eq.row_num(), self.LS_ineq.row_num()])

        A_upper = np.concatenate((A_m1_n1, A_m1_n2, neg_A_m1_n2, neg_E_m1_m1), axis=1)
        A_lower = np.concatenate((A_m2_n1, A_m2_n2, neg_A_m2_n2, O_m2_m1), axis=1)

        A = np.concatenate((A_upper, A_lower), axis=0)

        b = np.concatenate((self.LS_ineq.b.copy(), self.LS_eq.b.copy()), axis=0)

        c_n1 = self.c_objective[self.x_positive_indexes].copy()
        c_n2 = self.c_objective[x_any_sign_indexes].copy()
        neg_c_n2 = c_n2.copy() * -1
        o_m1 = np.zeros(self.LS_ineq.row_num())

        c = np.concatenate((c_n1, c_n2, neg_c_n2, o_m1), axis=0)

        x_new_size = A.shape[1]
        res = LPProblem(x_size=x_new_size,
                        LS_eq=LinearSystem(A, b, relationship='='),
                        LS_ineq=None,
                        x_positive_indexes=list(range(x_new_size)),
                        c_objective=c)

        if inplace:
            self.__dict__.update(res.__dict__)

        return res
