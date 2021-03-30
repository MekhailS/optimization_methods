from iterative_optimizer import IterativeOptimizer

import numpy as np
import scipy as sc


class GradientDescent2Order(IterativeOptimizer):

    def __init__(self, func_differentiable, dim):
        super().__init__(func_differentiable, dim)

    @property
    def __initial_x(self):
        return np.zeros(self._dim)

    def optimize(self, tol, step=1.0, print_info=True):
        """
            a := step

            x_{k+1} = x_{k} - a H^(-1)(x_{k}) * grad_f(x_{k})
            =>
            H(x_{k})*(x_{k+1}-x_{k}) = -a grad_f(x_{k})
            =>
            let z be solution of
            H(x_{k})*z = -a grad_f(x_{k})
            =>
            x_{k+1} = z + x_{k}
        """
        termination_rule = IterativeOptimizer.TerminationRule(
            optimizer=self,
            limit_iterations_termination_true=5,
            tol=tol,
            rule_name='grad_f'
        )

        x_cur = self.__initial_x
        x_history = []
        while True:
            if print_info:
                print(f'x is {x_cur}, function at x is {self._func.ev_func(x_cur)}')

            x_history.append(x_cur)
            grad, hessian_matrix = self._func.ev_gradient(x_cur), self._func.ev_hessian(x_cur)

            b = -step * grad
            """
                cholesky decomposition or method of square root
                find decomposition of hessian matrix as:
                
                H = LDL^t
                
                - where L is lower triangular matrix
                - D is diagonal matrix
                
                then solve the system LDL^T * z = b easily
            """
            z = sc.linalg.solve(
                a=hessian_matrix,
                b=b,
                assume_a='sym'
            )
            x_next = z + x_cur

            if termination_rule(x_cur, x_next):
                x_history.append(x_next)
                return x_next, x_history

            x_cur = x_next