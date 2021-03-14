
class OneDimOptimizer:
    def __init__(self, interval, func_obj):
        if interval[0] > interval[1]:
            raise ValueError('not valid interval')

        self._a = float(interval[0])
        self._b = float(interval[1])
        self._func_obj = func_obj

    def get_minimum_point(self, tol, print_iterations_info=False):
        pass
