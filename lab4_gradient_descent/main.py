import numpy as np
import scipy.optimize as opt

from gradient_steepest_descent import GradientSteepestDescent
from gradient_descent_second_order import GradientDescent2Order
from function_differentiable import FunctionDifferentiable
from dfp_method import DFP

from research import *

from call_count import call_count


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

F_DIM = 2
func_differentiable = FunctionDifferentiable(F_DIM, func, func_grad, func_hessian)

RESEARCH_PRINT_SEPARATOR = '------------------------'
RESEARCH_BIG_PRINT_SEPARATOR = '\n'


def research_on_orthogonality_of_curve_segments(optimizer, name):
    tol = 1.e-4
    _, x_history = optimizer.optimize(tol, print_info=False)
    curve_segments_pair_dot_products = orthogonality_of_curve_segments(x_history)

    print(f'orthogonality of curve segments for {name} method (numeration from 0) tol is {tol}')
    print(RESEARCH_PRINT_SEPARATOR)
    for i, dot_product in enumerate(curve_segments_pair_dot_products):
        print(f'dot product of {i+2}-th and {i+1}-th curve segments: {dot_product}')


def solve_problem(optimizer, tol_list, name=''):
    for tol in tol_list:
        x, _ = optimizer.optimize(tol, print_info=False)
        print(f'{name} solution is {x}; tolerance: {tol}')


def research_on_calls_of_function(optimizer, tol, name):
    call_count.zero_all_counts()

    optimizer.optimize(tol, print_info=False)
    print(f'call counts for {name} method : {call_count.all_counts()} tol is {tol}')

    call_count.zero_all_counts()


def draw_gradient_curve(optimizer, optimizer_name):
    tol = 1.e-4
    _, x_path = optimizer.optimize(tol, print_info=False)

    plot_contour_with_curves(
        func=func,
        curve_x=x_path,
        curve_name=f'{optimizer_name} full path'
    )
    plot_contour_with_curves(
        func=func,
        curve_x=x_path[1:],
        curve_name=f'{optimizer_name} starting from 2nd point'
    )


def main():
    optimizers_dict = {
        'Gradient steepest descent': GradientSteepestDescent(func_differentiable),
        'Newton gradient descent': GradientDescent2Order(func_differentiable),
        'Davidon–Fletcher–Powell method': DFP(func_differentiable)
    }

    tol_list = [1.e-1, 1.e-3, 1.e-4]
    for optimizer_name, optimizer in optimizers_dict.items():
        print(RESEARCH_PRINT_SEPARATOR)
        solve_problem(optimizer, tol_list, optimizer_name)

    print(RESEARCH_BIG_PRINT_SEPARATOR)
    tol_for_call_count = 1.e-4
    for optimizer_name, optimizer in optimizers_dict.items():
        print(RESEARCH_PRINT_SEPARATOR)
        research_on_calls_of_function(optimizer, tol_for_call_count, optimizer_name)

    print(RESEARCH_BIG_PRINT_SEPARATOR)
    method_name = 'Gradient steepest descent'
    research_on_orthogonality_of_curve_segments(
        optimizers_dict[method_name], method_name
    )

    for optimizer_name, optimizer in optimizers_dict.items():
        draw_gradient_curve(optimizer, optimizer_name)


main()