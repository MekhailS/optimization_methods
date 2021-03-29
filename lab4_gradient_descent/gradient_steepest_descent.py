import numpy as np


class GradientSteepestDescent:

    def __init__(self, func_differentiable, dim):
        self.__dim = dim
        self.__func = func_differentiable

    @property
    def __initial_x(self):
        return np.zeros(self.__dim)

    def optimize(self, tol):
        n_iterations_func_not_changing = 0
        LIMIT_ITERATIONS_FUNC_NOT_CHANGING = 5

        def stopping_rule(x_cur, x_next):
            nonlocal n_iterations_func_not_changing

            if np.abs(self.__func.ev_func(x_cur) - self.__func.ev_func(x_next)) < tol:
                n_iterations_func_not_changing += 1
            else:
                n_iterations_func_not_changing = 0

            if n_iterations_func_not_changing >= LIMIT_ITERATIONS_FUNC_NOT_CHANGING:
                return True
            return False

        x_history = []
        x_cur = self.__initial_x
        while True:
            x_history.append(x_cur)

            def func_after_step(step):
                return self.__func(self.__func.ev_func(x_cur) - step * self.__func.ev_gradient(x_cur))

            step = 0.01
            x_next = self.__func.ev_func(x_cur) - step * self.__func.ev_gradient(x_cur)

            if stopping_rule(x_cur, x_next):
                x_history.append(x_next)
                return x_next, x_history

            x_cur = x_next
