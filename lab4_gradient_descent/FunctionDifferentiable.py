

class FunctionDifferentiable:

    def __init__(self, func, gradient):
        """
        :param func: function with one argument as n-dimensional numpy array
        :param gradient: gradient of function 'func' with one argument as n-dimensional numpy array
        """
        self.__func = func
        self.__gradient = func

    @property
    def ev_func(self, x):
        return self.__func(x)

    @property
    def ev_gradient(self, x):
        return self.__gradient(x)