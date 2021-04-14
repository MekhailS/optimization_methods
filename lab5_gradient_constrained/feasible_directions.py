import numpy as np
from scipy.optimize import linprog

from function_differentiable import FunctionDifferentiable


class FeasibleDirectionsOptimizer:

    def __init__(self, func_obj, phi_ineq_list):
        self.func_obj = func_obj
        self.phi_ineq_list = phi_ineq_list

    def optimize(self):
        # zero step problem
        theta_obj, phi_ineq_list = self.__zero_step_subtask()
        zero_step_problem = FeasibleDirectionsOptimizer(theta_obj, phi_ineq_list)
        x0_w_theta = zero_step_problem.__inner_optimize(np.zeros(theta_obj.dim), 1.0)
        x_0, theta = x0_w_theta[:-1], x0_w_theta[-1]

        # result
        return self.__inner_optimize(x_0, -theta)

    @staticmethod
    def __generate_phi_zero_step_subtask(phi):
        phi_func = lambda x: phi.ev_func(x[:-1]) - x[-1]
        def grad_phi(x):
            return np.append(phi.ev_gradient(x[:-1]), -1.0)

        return FunctionDifferentiable(
            dim=phi.dim + 1,
            func=phi_func,
            gradient=grad_phi
        )

    def __zero_step_subtask(self):
        objective_theta = lambda x: x[-1]
        def grad_objective_theta(x):
            grad = np.zeros(len(x))
            grad[-1] = 1.0
            return grad

        func_theta = FunctionDifferentiable(
            dim=self.phi_ineq_list[0].dim + 1,
            func=objective_theta,
            gradient=grad_objective_theta
        )
        phi_new_list = [
            self.__generate_phi_zero_step_subtask(phi)
            for phi in self.phi_ineq_list
        ]

        return func_theta, phi_new_list

    def __inner_optimize(self, x_0, delta_0):
        ZERO_TOL = 1.e-7
        lambda_ = 0.5
        xi_list = np.ones(len(self.phi_ineq_list) + 1)

        x_cur, theta_cur, delta_cur = x_0, None, delta_0

        step_count = 0
        while True:
            s_cur, theta_cur = self.__first_problem(x_cur, xi_list, delta_cur)

            if step_count == 0:
                delta_cur = -theta_cur

            if -delta_cur <= theta_cur < 0:
                x_cur = x_cur
                delta_cur *= lambda_

            elif theta_cur < -delta_cur:
                alpha_k = self.__find_alpha_cur(x_cur, theta_cur, s_cur, xi_list, lambda_)
                x_cur += alpha_k * s_cur
                delta_cur = delta_cur

            delta_ok = - max(
                (self.func_obj.ev_func(x_cur),
                 max(
                     [phi.ev_func(x_cur) for phi in self.phi_ineq_list
                      if phi.ev_func(x_cur) != 0]
                 )
                 )
            )
            step_count += 1
            if delta_cur < delta_ok and np.abs(theta_cur) <= ZERO_TOL:
                return x_cur

    def __find_alpha_cur(self, x_cur, theta_cur, s_cur, xi_list, lambda_):
        alpha_cur = 1.0
        while not (
                (self.func_obj.ev_func(x_cur + alpha_cur*s_cur) - self.func_obj.ev_func(x_cur) \
                <= 0.5 * xi_list[0] * theta_cur * alpha_cur)
            and
                (np.asarray([phi.ev_func(x_cur + alpha_cur * s_cur) <= 0 for
                             phi in self.phi_ineq_list])
                ).all()
        ):
            shit_2 = (self.func_obj.ev_func(x_cur + alpha_cur*s_cur) - self.func_obj.ev_func(x_cur))
            shit_3 = 0.5 * xi_list[0] * theta_cur * alpha_cur
            shit = np.asarray([phi.ev_func(x_cur + alpha_cur * s_cur) <= 0 for
                               phi in self.phi_ineq_list])
            alpha_cur *= lambda_
        return alpha_cur

    def __first_problem(self, x_cur, xi_list, delta):
        # construct matrix for first subtask
        A_matrix_simplex = [
            phi_i.ev_gradient(x_cur)
            for phi_i in self.phi_ineq_list
        ]
        A_matrix_simplex = np.column_stack(
            [A_matrix_simplex, xi_list[1:].reshape((-1, 1))]
        )
        J = [-delta <= phi_i.ev_func(x_cur) <= 0 for phi_i in self.phi_ineq_list]

        A_matrix_simplex = np.row_stack(
            [A_matrix_simplex,
             np.append(self.func_obj.ev_gradient(x_cur), xi_list[0])]
        )
        A_matrix_simplex = A_matrix_simplex[np.append(J, True)]

        # b vector
        b_simplex = np.zeros(len(A_matrix_simplex))

        # bounds
        bounds_simplex = [(-1, 1) for _ in range(len(x_cur))]
        bounds_simplex.append((None, None))

        # objective function
        c_simplex = np.append(np.zeros(self.func_obj.dim), 1.0)

        # solve using simplex
        res = linprog(
            c=c_simplex,
            A_ub=A_matrix_simplex,
            b_ub=b_simplex,
            bounds=bounds_simplex,
            method='simplex'
        )
        x_res = res.x
        return np.asarray(x_res[:-1]), x_res[-1]

