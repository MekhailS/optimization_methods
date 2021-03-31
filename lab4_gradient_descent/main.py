import numpy as np
import scipy.optimize as opt

from gradient_steepest_descent import GradientSteepestDescent
from gradient_descent_second_order import GradientDescent2Order
from function_differentiable import FunctionDifferentiable
from dfp_method import DFP
from call_count import call_count


if __name__ == '__main__':

    a, b, c = 2.0, 5.0, 3.0

    def func(x):
        return a*x[0] + x[1] + 4.0*np.sqrt(1.0 + b*x[0]**2 + c*x[1]**2)

    def func_grad(x):
        return [
            a + 4.0*b*x[0]/np.sqrt(b*x[0]**2 + c*x[1]**2 + 1.0),
            1.0 + 4.0*c*x[1]/np.sqrt(b*x[0]**2 + c*x[1]**2 + 1.0),
        ]

    def func_hessian(x):
        x, y = x[0], x[1]
        sqrt_denominator = np.sqrt(b*x**2 + c*y**2 + 1.0)
        return [
            [4.0*b/sqrt_denominator - 4*(b**2)*(x**2)/sqrt_denominator**3, -4*b*c*x*y/sqrt_denominator**3],
            [-4*b*c*x*y/sqrt_denominator**3, 4.0*c/sqrt_denominator - 4*(c**2)*(y**2)/sqrt_denominator**3]
        ]

    tol = 1.e-7

    function_differentiable = FunctionDifferentiable(func, func_grad, func_hessian)

    gds = GradientSteepestDescent(function_differentiable, 2)
    gd2ord = GradientDescent2Order(function_differentiable, 2)
    dfp = DFP(function_differentiable, 2)

    call_count.zero_all_counts()
    res_gds = gds.optimize(tol)

    print(call_count.all_counts())

    call_count.zero_all_counts()
    res_gd2ord = gd2ord.optimize(tol)

    print(call_count.all_counts())


    call_count.zero_all_counts()
    res = dfp.optimize(tol)

    print(call_count.all_counts())

    res_scipy = opt.minimize(func, [0, 0], tol=tol)

    print('lab4')