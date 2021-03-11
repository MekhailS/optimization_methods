from lab3_one_dimensional_unconstrained.one_dim_optimizer import OneDimOptimizer


class BisectionMethodOptimizer(OneDimOptimizer):

    DELTA_MIDPOINT_FRACTION = 0.001

    def __init__(self, interval, func_obj):
        super().__init__(interval, func_obj)

        self.__a = interval[0]
        self.__b = interval[1]
        self.__func_obj = func_obj

    @property
    def __interval_len(self):
        return abs(self.__b - self.__a)

    @property
    def __interval_mid_point(self):
        return (self.__b + self.__a) / 2

    def get_minimum_point(self, tol):
        a_backup, b_backup = self.__a, self.__b
        while self.__interval_len > tol:
            x_mid = self.__interval_mid_point
            delta = self.__interval_len * self.DELTA_MIDPOINT_FRACTION

            x_mid_left, x_mid_right = x_mid - delta, x_mid + delta
            f_mid_left, f_mid_right = self.__func_obj(x_mid_left), self.__func_obj(x_mid_right)

            if f_mid_left > f_mid_right:
                self.__a, self.__b = x_mid_left, self.__b
            elif f_mid_left == f_mid_right:
                self.__a, self.__b = x_mid_left, x_mid_right
            else:
                self.__a, self.__b = self.__a, x_mid_right

        res = self.__interval_mid_point
        self.__a, self.__b = a_backup, b_backup
        return res
