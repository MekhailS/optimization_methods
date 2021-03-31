from functools import lru_cache


class FibonacchiSolver:
    def __init__(self, target_func, a, b, tol=1.e-3):
        self._target_func = target_func
        self._a = a
        self._b = b
        self._tol = tol
        self._N = 1
        self._find_N()

    @lru_cache()
    def _fib(self, n):
        return n if n < 2 else self._fib(n - 1) + self._fib(n - 2)

    def _x_1(self, i):
        return (self._fib(self._N - i - 1) / self._fib(self._N - i + 1)) * \
               (self._b - self._a) + self._a

    def _x_2(self, i):
        return (self._fib(self._N - i) / self._fib(self._N - i + 1)) * \
               (self._b - self._a) + self._a

    def _find_N(self):
        while self._fib(self._N) <= (self._b - self._a) / self._tol:
            self._N += 1

    def solve(self):
        x_1 = self._x_1(1)
        x_2 = self._x_2(1)

        for i in range(1, self._N - 2):
            if self._target_func(x_2) > self._target_func(x_1):
                self._b = x_2
                x_2 = x_1
                x_1 = self._x_1(i)
            else:
                self._a = x_1
                x_1 = x_2
                x_2 = self._x_2(i)

        return (self._a + self._b) / 2
