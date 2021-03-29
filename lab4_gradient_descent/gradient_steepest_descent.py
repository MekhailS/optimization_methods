import numpy as np
from scipy import optimize as opt
from fibonacchi import FibonacchiSolver


class GradientSteepestDescent:

    def __init__(self, func_differentiable, dim):
        self.__dim = dim
        self.__func = func_differentiable

    @property
    def __initial_x(self):
        #return np.zeros(self.__dim)
        return np.random.randn(self.__dim)

    def optimize(self, tol):
        n_iterations_func_not_changing = 0
        LIMIT_ITERATIONS_FUNC_NOT_CHANGING = 10

        def stopping_rule(x_cur, x_next, rule='df'):
            nonlocal n_iterations_func_not_changing

            dict_rule = {
                'df':
                    lambda x_cur, x_next : np.abs(self.__func.ev_func(x_cur) - self.__func.ev_func(x_next)) ,
                'dx':
                    lambda x_cur, x_next : np.linalg.norm(x_cur - x_next, ord=2),
                'grad_f':
                    lambda x_cur, _ : np.linalg.norm(self.__func.ev_gradient(x_next), ord=2)
            }

            if dict_rule[rule](x_cur, x_next) <= tol:
                n_iterations_func_not_changing += 1
            else:
                n_iterations_func_not_changing = 0

            if n_iterations_func_not_changing >= LIMIT_ITERATIONS_FUNC_NOT_CHANGING:
                return True
            return False

        x_history = []
        x_cur = self.__initial_x
        while True:
            print(f'x is {x_cur}, function at x is {self.__func.ev_func(x_cur)}')
            x_history.append(x_cur)

            grad_x_cur = self.__func.ev_gradient(x_cur)
            def func_after_step(step):
                x_next = x_cur - step * grad_x_cur
                return self.__func.ev_func(x_next)

            STEP_RANGE = [0, 1]
            fib_solver = FibonacchiSolver(
                func_after_step,
                STEP_RANGE[0], STEP_RANGE[1],
                result_interval_length=1.e-3,
                eps=1.e-10
            )
            step = fib_solver.solve()
            x_next = x_cur - step * grad_x_cur

            if stopping_rule(x_cur, x_next, rule='grad_f'):
                x_history.append(x_next)
                return x_next, x_history

            x_cur = x_next
