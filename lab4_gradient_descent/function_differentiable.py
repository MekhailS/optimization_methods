import numpy as np

from call_count import call_count


class FunctionDifferentiable:

    def __init__(self, func, gradient=None, hessian=None):
        """
        :param func: function with one argument as n-dimensional numpy array
        :param gradient: gradient of function 'func' with one argument as n-dimensional numpy array
        """
        self.__func = func
        self.__gradient = gradient
        self.__hessian = hessian

    @call_count
    def ev_func(self, x):
        x = np.asarray(x)
        return self.__func(x)

    @call_count
    def ev_gradient(self, x):
        if self.__gradient is None:
            return None

        x = np.asarray(x)
        return np.asarray(self.__gradient(x))

    @call_count
    def ev_hessian(self, x):
        if self.__hessian is None:
            return None

        x = np.asarray(x)
        return np.asarray(self.__hessian(x))