from fibonacchi import FibonacchiSolver
from iterative_optimizer import IterativeOptimizer

import numpy as np


class GradientSteepestDescent(IterativeOptimizer):

    def __init__(self, func_differentiable, dim):
        super().__init__(func_differentiable, dim)

    @property
    def __initial_x(self):
        return np.random.randn(self._dim)

    def optimize(self, tol, print_info=True):
        termination_rule = IterativeOptimizer.TerminationRule(
            optimizer=self,
            limit_iterations_termination_true=10,
            tol=tol,
            rule_name='grad_f'
        )

        x_history = []
        x_cur = self.__initial_x
        while True:
            if print_info:
                print(f'x is {x_cur}, function at x is {self._func.ev_func(x_cur)}')

            x_history.append(x_cur)

            grad_x_cur = self._func.ev_gradient(x_cur)
            def func_after_step(step):
                x_next = x_cur - step * grad_x_cur
                return self._func.ev_func(x_next)

            STEP_RANGE = [0, 1]
            fib_solver = FibonacchiSolver(
                func_after_step,
                STEP_RANGE[0], STEP_RANGE[1],
                result_interval_length=1.e-3,
                eps=1.e-10
            )
            step = fib_solver.solve()
            x_next = x_cur - step * grad_x_cur

            if termination_rule(x_cur, x_next):
                x_history.append(x_next)
                return x_next, x_history

            x_cur = x_next
