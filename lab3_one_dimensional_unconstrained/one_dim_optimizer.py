
class OneDimOptimizer:
    def __init__(self, interval, func_obj):
        if interval[0] > interval[1]:
            raise ValueError('not valid interval')

    def get_minimum_point(self, tol):
        pass
