from lab3_one_dimensional_unconstrained.func_obj import FuncObj
from lab3_one_dimensional_unconstrained.bisection_method_optimizer import BisectionMethodOptimizer
from lab3_one_dimensional_unconstrained.quadratic_approx_optimizer import QuadraticApproxOptimizer


def f_line_2p(x1, y1, x2, y2):
    return lambda x: y1 + (y2-y1)/(x2-x1)*(x-x1)


if __name__ == '__main__':
    f_problem = lambda x: 3 * x ** 4 - 10 * x ** 3 + 21 * x ** 2 + 12 * x

    '''
    def f_problem_modified(x):
        if 0 <= x <= 0.1:
            return f_line_2p(0, f_problem(0), 0.1, -10)(x)
        if 0.1 <= x <= 0.25:
            return f_line_2p(0.1, -10, 0.25, f_problem(0.25))(x)
        else:
            return f_problem(x)
    '''


    tol_problem = 1.e-6
    interval_problem = [-0.5, 0.5]

    f_1 = lambda x: -x**3 - 0.2*x**2 + 2.4*x + 1
    tol_1 = 0.001
    interval_1 = [-10, -1.5]

    func_obj_quadratic = FuncObj(f_problem)
    tol = tol_problem
    interval = interval_problem

    bisection_optimizer = BisectionMethodOptimizer(interval, func_obj_quadratic)
    print(f'BISECTION: {bisection_optimizer.get_minimum_point(tol, print_iterations_info=True)}')

    print(func_obj_quadratic.__dict__)
    func_obj_quadratic.zero_call_count()

    quadratic_approx_optimizer = QuadraticApproxOptimizer(interval, func_obj_quadratic)
    print(f'QUADRATIC APPROX: {quadratic_approx_optimizer.get_minimum_point(tol, print_iterations_info=True)}')

    print(func_obj_quadratic.__dict__)