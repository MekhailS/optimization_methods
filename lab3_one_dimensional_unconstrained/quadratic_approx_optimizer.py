from lab3_one_dimensional_unconstrained.one_dim_optimizer import OneDimOptimizer


class QuadraticApproxOptimizer(OneDimOptimizer):

    def __init__(self, interval, func_obj):
        super().__init__(interval, func_obj)

    def get_minimum_point(self, tol, print_iterations_info=False):
        x = [self._a, (self._a + self._b)/2, self._b]
        y = [self._func_obj(x_el) for x_el in x]

        x_prev = x[1]
        while True:
            if print_iterations_info:
                print(f'current interval: [{x[0]}, {x[2]}]; point inside: {x[1]}')

            if x[0] >= self._b:
                return self._b
            if x[2] <= self._a:
                return self._a

            a = [
                y[0],
                (y[1] - y[0])/(x[1] - x[0]),
                1/(x[2] - x[1]) * ((y[2] - y[0])/(x[2] - x[0]) - (y[1] - y[0])/(x[1] - x[0]))
            ]
            x_star = 1/2 * (x[1] + x[0] - a[1]/a[2])
            y_star = self._func_obj(x_star)

            if abs(x_star - x_prev) < tol:
                return x_star

            x_prev = x_star
            if x[0] <= x_star <= x[2]:
                # assume x_star < x[1]:
                x_left, x_right = x_star, x[1]
                y_left, y_right = y_star, y[1]
                if x_left > x_right:
                    # otherwise, swap
                    x_left, x_right = x_right, x_left
                    y_left, y_right = y_right, y_left

                '''
                order of x points:
                x[0] < x_left < x_right < x[2]
                '''

                # pick new 3 points for next iteration
                if y_left < y_right:
                    x = [x[0], x_left, x_right]
                    y = [y[0], y_left, y_right]
                else:
                    x = [x_left, x_right, x[2]]
                    y = [y_left, y_right, y[2]]

            elif x_star > x[2]:
                x = [x[1], x[2], x_star]
                y = [y[1], y[2], y_star]

            elif x_star < x[0]:
                x = [x_star, x[0], x[1]]
                y = [y_star, x[0], x[1]]

