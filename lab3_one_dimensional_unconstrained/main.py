from lab3_one_dimensional_unconstrained.func_obj import FuncObj
from lab3_one_dimensional_unconstrained.bisection_method_optimizer import BisectionMethodOptimizer
from lab3_one_dimensional_unconstrained.quadratic_approx_optimizer import QuadraticApproxOptimizer

if __name__ == '__main__':
    f_problem = lambda x: 3 * x ** 4 - 10 * x ** 3 + 21 * x ** 2 + 12 * x
    tol_problem = 0.01
    interval_problem = [0, 0.5]

    f_1 = lambda x: -x**3 - 0.2*x**2 + 2.4*x + 1
    tol_1 = 0.001
    interval_1 = [-2.5, -1.5]

    func_obj_quadratic = FuncObj(f_1)
    tol = tol_1
    interval = interval_1

    bisection_optimizer = BisectionMethodOptimizer(interval, func_obj_quadratic)
    print(f'BISECTION: {bisection_optimizer.get_minimum_point(tol)}')

    print(func_obj_quadratic.__dict__)
    func_obj_quadratic.zero_call_count()

    quadratic_approx_optimizer = QuadraticApproxOptimizer(interval, func_obj_quadratic)
    print(f'QUADRATIC APPROX: {quadratic_approx_optimizer.get_minimum_point(tol)}')

    print(func_obj_quadratic.__dict__)