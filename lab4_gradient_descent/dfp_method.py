import numpy as np
from scipy import optimize
from iterative_optimizer import IterativeOptimizer

from fibonacchi import FibonacchiSolver

# Davidon–Fletcher–Powell method
class DFP(IterativeOptimizer):

    def __init__(self, func_differentiable):
        super().__init__(func_differentiable)
        self.__termination_rule = None
        self.__A = np.identity(self._func.dim)
        self.__cur_index = 1
        self.__x_history = []

        x_0 = np.zeros(self._func.dim)
        self.__x_history.append(x_0)

        self.step = 'first'

    def __second_step(self):
        p_k = self.__A @ (-self._func.ev_gradient(self.__x_history[-1]))

        alpha_k = FibonacchiSolver(
            lambda alpha: self._func.ev_func(self.__x_history[-1] + alpha * p_k),
            0,
            1,
        ).solve()

        self.__x_history.append(self.__x_history[-1] + alpha_k * p_k)

        dim = self._func.dim
        if self.__cur_index == dim or (self.__cur_index / dim) % 2 == 0:
            self.__A = np.identity(dim)
            self.__cur_index += 1

            self.step = 'first'
        else:
            self.step = 'third'

    def __third_step(self):
        delta_x = np.array(self.__x_history[-1] - self.__x_history[-2])
        delta_omega = np.array(
            self._func.ev_gradient(self.__x_history[-2]) - self._func.ev_gradient(self.__x_history[-1]))

        delta_x = delta_x.reshape((len(delta_x)), 1)
        delta_omega = delta_omega.reshape((len(delta_omega)), 1)

        _B = np.dot(delta_x, delta_x.T) / (np.transpose(delta_omega) @ delta_x)
        _C = (self.__A @ delta_omega @ np.transpose(delta_omega) @ np.transpose(self.__A)) / (
                np.transpose(delta_omega) @ self.__A @ delta_omega)
        self.__A = self.__A - _B - _C

        self.__cur_index += 1

        self.step = 'first'

    def optimize(self, tol, print_info=True):
        self.__termination_rule = IterativeOptimizer.TerminationRule(
            optimizer=self,
            limit_iterations_termination_true=1,
            tol=tol,
            rule_name='grad_f'
        )

        # result = self.__first_step()
        while True:
            if self.step == 'first':
                if self.__termination_rule(0, self.__x_history[-1]):
                    result = self.__x_history[-1], self.__x_history
                    break
                else:
                    self.step = 'second'

            elif self.step == 'second':
                self.__second_step()

            elif self.step == 'third':
                self.__third_step()

        if print_info:
            print('DAVIDON-FLETCHER-POWELL METHOD:')
            print(f'optimal solution is {result[0]}, function at x is {self._func.ev_func(result[0])}')

            print('Step by step solution:')
            for x in result[1]:
                print(f'x is {x}, function at x is {self._func.ev_func(x)}')

        return result
