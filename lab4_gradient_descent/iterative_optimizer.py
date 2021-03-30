import numpy as np


class IterativeOptimizer:
    """
    Base class for iterative unconstrained optimizers like gradient descent.
    Attributes:
        _func   objective function (FunctionDifferentiable instance)
        _dim    dimension of _func input (int)
        _dict_termination_rule  dictionary with keys as names for termination rules (stopping condition)
                                and keys as functions of signature f(x_cur, x_next) and return value float
                                they're being evaluated to check if stopping condition is true

        !!! you CAN add custom termination rules to _dict_termination_rule after inheritance !!!
    """

    def __init__(self, func_differentiable, dim):
        """
        initialize iterative optimizer
        :param func_differentiable: objective function FunctionDifferentiable instance
        :param dim: dimension of function input
        """

        self._dim = dim
        self._func = func_differentiable

        self._dict_termination_rule = {
            'df':
                lambda x_cur, x_next: np.abs(self._func.ev_func(x_cur) - self._func.ev_func(x_next)),
            'dx':
                lambda x_cur, x_next: np.linalg.norm(x_cur - x_next, ord=2),
            'grad_f':
                lambda _, x_next: np.linalg.norm(self._func.ev_gradient(x_next), ord=2)
        }

    class TerminationRule:
        """
        Class which encapsulates evaluation of termination rule of given optimizer

        !!! to create termination rule linked with your optimizer, construct instance of this class
        with argument 'optimizer=self' !!!

        usage example can be seen in 'gradient_steepest_descent.py', function 'optimize'
        """

        def __init__(self, optimizer, limit_iterations_termination_true, tol, rule_name):
            """
            initialize termination rule instance with given optimizer
            :param optimizer: optimizer to be linked with (uses its _dict_termination_rule)
                              (instance inherited from IterativeOptimizer)
                              to link with your optimizer, set it to 'self'
            :param limit_iterations_termination_true: if the number of true evaluations (in a row)
                                                      of termination rule's func == this value,
                                                      True is returned on __call_
            :param tol: tolerance of termination rule's func
            :param rule_name: key name of termination rule's func from optimizer._dict_termination_rule
            """

            self._n_iterations_succeeds = 0
            self._limit_iterations_termination_true = limit_iterations_termination_true

            self.tol = tol

            self._term_rule = optimizer._dict_termination_rule[rule_name]

        def __call__(self, x_cur, x_next):
            """
            evaluate termination rule with x_k := x_cur, x_k+1 := x_next
            :param x_cur: x_k
            :param x_next: x_k+1
            :return: True if number of number of successive calls of termination rule function in a row
                    _limit_iterations_termination_true
            """

            if self._term_rule(x_cur, x_next) <= self.tol:
                self._n_iterations_succeeds += 1
            else:
                self._n_iterations_succeeds = 0

            if self._n_iterations_succeeds >= self._limit_iterations_termination_true:
                self._n_iterations_succeeds = 0
                return True
            return False

