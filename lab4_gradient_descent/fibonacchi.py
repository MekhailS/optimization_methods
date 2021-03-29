from linecache import cache


class FibonacchiSolver:
    def __init__(self, target_func, a, b, result_interval_length=0.001, eps=0.001):
        self._target_func = target_func
        self._a = a
        self._b = b
        self._result_interval_length = result_interval_length
        self._eps = eps
        self._preparation()

    @cache
    def _fib(self, n):
        return n if n < 2 else self.fib(n - 1) + self.fib(n - 2)

    def _preparation(self):
        self._N = 1
        self._k = 1
        while (self._fib(self._N) <= (self._b - self._a) / self._result_interval_length):
            self._N += 1
        self._lmbd = self._a + self._fib(self._N - 2) / self._fib(self._N) * (self._b - self._a)
        self._mu = self._a + self._fib(self._N - 1) / self._fib(self._N) * (self._b - self._a)

    def _first_step(self):
        if self._target_func(self._lmbd) > self._target_func(self._mu):
            return self._second_step()
        else:
            return self._third_step()

    def _second_step(self):
        self._a = self._lmbd
        self._lmbd = self._mu
        self._mu = self._a + self._fib(self._N - self._k - 1) / self._fib(self._N - self._k) * \
                   (self._b - self._a)
        if self._k == self._N - 2:
            return self._fifth_step()
        else:
            return self._fourth_step()

    def _third_step(self):
        self._b = self._mu
        self._mu = self._lmbd
        self._lmbd = self._a + self._fib(self._N - self._k - 2) / self._fib(self._N - self._k) * \
                     (self._b - self._a)
        if self._k == self._N - 2:
            return self._fifth_step()
        else:
            return self._fourth_step()

    def _fourth_step(self):
        self._k += 1
        return self._first_step()

    def _fifth_step(self):
        self._mu = self._lmbd + self._eps
        if self._target_func(self._lmbd) == self._target_func(self._mu):
            self._a = self._lmbd
        elif self._target_func(self._lmbd) < self._target_func(self._mu):
            self._b = self._mu
        return (self._b - self._a) / 2

    def solve(self):
        return self._first_step()
