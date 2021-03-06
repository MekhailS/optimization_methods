import numpy as np

from function_differentiable import FunctionDifferentiable
from feasible_directions import FeasibleDirectionsOptimizer

a, b, c = 2.0, 5.0, 3.0

def func(x):
    return a*x[0] + x[1] + 4.0*np.sqrt(1.0 + b*x[0]**2 + c*x[1]**2)

def func_grad(x):
    return [
        a + 4.0*b*x[0]/np.sqrt(b*x[0]**2 + c*x[1]**2 + 1.0),
        1.0 + 4.0*c*x[1]/np.sqrt(b*x[0]**2 + c*x[1]**2 + 1.0),
    ]

def coefs():
    return 1, 4, 2


def f(x):
    a, b, c = coefs()
    return a * x[0] + x[1] + 4 * np.sqrt(1 + b * x[0] ** 2 + c * x[1] ** 2)


def grad_f(x):
    a, b, c = coefs()
    return np.asarray([a + 4 * b * x[0] / np.sqrt(1 + b * x[0] ** 2 + c * x[1] ** 2),
    1 + 4 * c * x[1] / np.sqrt(1 + b * x[0] ** 2 + c * x[1] ** 2)])

F_DIM = 2
func_differentiable = FunctionDifferentiable(F_DIM, func, func_grad)

phi_functions = [
    lambda x: -x[0] + x[1] - 5,
    lambda x: -3.0*x[0] - x[1] - 15.0,
    lambda x: 1.5*x[0] - x[1] - 5.0,
    lambda x: 4.0*x[0] + x[1] - 20
]

phi_functions_quadratic = [
    lambda x: -0.2*x[0]**2 + x[1] - 3,
    lambda x: x[0]**2 - x[1] - 5
]

phi_gradients = [
    lambda x: [-1.0, 1],
    lambda x: [-3.0, -1.0],
    lambda x: [1.5, -1.0],
    lambda x: [-4.0, 1.0]
]

phi_gradients_quadratic = [
    lambda x: [-0.4*x[0], 1],
    lambda x: [2*x[0], -1]
]

funs = [
    lambda x: 2 * x[0] + 5 * x[1],
    lambda x: -x[0] - x[1] - 1,
    lambda x: -3 * x[0] + 2 * x[1] - 1,
    lambda x: -9 * x[0] + 4 * x[1] + 1,
    lambda x: x[0],
    lambda x: x[1],
]
funs_grads = [
    lambda x: np.array([2, 5]),
    lambda x: np.array([-1, -1]),
    lambda x: np.array([-3, 2]),
    lambda x: np.array([-9, 4]),
    lambda x: np.array([1, 0]),
    lambda x: np.array([0, 1]),
]

phi_ineq_list = [FunctionDifferentiable(F_DIM, func, grad)
                 for func, grad in zip(phi_functions, phi_gradients)]


def main():
    FDO = FeasibleDirectionsOptimizer(
        func_obj=func_differentiable,
        phi_ineq_list=phi_ineq_list
    )
    x_star, x_history = FDO.optimize(print_info=True)
    pass

if __name__ == '__main__':
    main()