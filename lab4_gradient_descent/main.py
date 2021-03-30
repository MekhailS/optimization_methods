import numpy as np
import scipy.optimize as opt

from gradient_steepest_descent import GradientSteepestDescent
from function_differentiable import FunctionDifferentiable
from call_count import call_count
from dfp_method import DFP

if __name__ == '__main__':

    a, b, c = 2.0, 5.0, 3.0

    i = 1
    def func(x):
        return a*x[0] + x[1] + 4.0*np.sqrt(1.0 + b*x[0]**2 + c*x[1]**2)

    def func_grad(x):
        return [
            a + 4.0*b*x[0]/np.sqrt(b*x[0]**2 + c*x[1]**2 + 1.0),
            1.0 + 4.0*c*x[1]/np.sqrt(b*x[0]**2 + c*x[1]**2 + 1.0),
        ]

    p = [-49.798, 48.202]
    print(f'shit = {func_grad(p)}')

    tol = 1.e-5


    function_differentiable = FunctionDifferentiable(func, func_grad)

    dfp = DFP(function_differentiable, 2)
    res = dfp.optimize(tol)


    gds = GradientSteepestDescent(function_differentiable, 2)

    res = gds.optimize(tol)
    # print(res)
    print(f'num iterations: {len(res[1])}')

    res_scipy = opt.minimize(func, [0, 0], tol=tol)

    print(call_count.all_counts())

    print('lab4')