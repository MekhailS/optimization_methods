from lab3_one_dimensional_unconstrained.one_dim_optimizer import OneDimOptimizer


class BisectionMethodOptimizer(OneDimOptimizer):

    DELTA_MIDPOINT_FRACTION = 0.001

    @property
    def __interval_len(self):
        return abs(self._b - self._a)

    @property
    def __interval_mid_point(self):
        return (self._b + self._a) / 2

    def get_minimum_point(self, tol, print_iterations_info=False):
        a_backup, b_backup = self._a, self._b
        while True:
            if print_iterations_info:
                print(f'current interval: [{self._a}, {self._b}]')

            if self.__interval_len <= tol:
                break

            x_mid = self.__interval_mid_point
            delta = self.__interval_len * self.DELTA_MIDPOINT_FRACTION

            x_mid_left, x_mid_right = x_mid - delta, x_mid + delta
            f_mid_left, f_mid_right = self._func_obj(x_mid_left), self._func_obj(x_mid_right)

            if f_mid_left > f_mid_right:
                self._a, self._b = x_mid_left, self._b
            elif f_mid_left == f_mid_right:
                self._a, self._b = x_mid_left, x_mid_right
            else:
                self._a, self._b = self._a, x_mid_right

        res = self.__interval_mid_point
        self._a, self._b = a_backup, b_backup
        return res
