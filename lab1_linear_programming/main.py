import numpy as np
import copy
from interface import *
from lab1_linear_programming.data_parser import *
from lab1_linear_programming.interface import *
from lab1_linear_programming.plot_function import visualize_func_in_2d_area
from lab1_linear_programming.linear_programming import *

def viktor_main():
    lp_problem = LPProblem([[5, 2, 3, 6, -2],
                            [10, 5, 6, 43, 10],
                            [6, -7, -2, 1, 4],
                            [1, 3, 4, 6, 8],
                            [4, 7, 3, 2, 1]],
                           [5, 10, 1, 1, 0],
                           [2, 3, 4, 1, 1],
                           [3, 4], [0, 1, 2], [1, 3], [0, 2, 4]
                           )
    lp_problem.from_common_to_dual(implace=True)
    lp_problem.from_common_to_dual(implace=True)

    lp_problem_canonical = LPProblem([[5, 2, 3, 6, -2],
                                      [10, 5, 6, 43, 10],
                                      [6, -7, -2, 1, 4],
                                      [4, 7, 3, 2, 1]],
                                     [5, 10, 1, 1],
                                     [2, 3, 4, 1, 1],
                                     M2_b_eq=[0, 1, 2, 3],
                                     N1_x_positive=[0, 1, 2, 3, 4]
                                     )
    t = lp_problem_canonical.from_common_to_dual()
    y = t.from_common_to_dual()

    visualize_func_in_2d_area(lp_problem.objective_func(),
                              lp_problem.optimization_area_indicator())
    t = lp_problem.dualLPProblem()
    y = t.dualLPProblem()
    visualize_func_in_2d_area(y.objective_func(),
                              y.optimization_area_indicator())
    lp_problem.canonical(inplace=True)


def mikhail_main():
    lp_problem_2d = LPProblem(
        x_dim=2,
        A=[[-1.0, -1.0],
           [-1.0, 1.0],
           [2.0, 1.0]],
        b=[-20.0, -20.0, -1.0],
        c_objective=[1.0, 1.0],
        M1_b_ineq=[0, 1, 2],
        N1_x_positive=None
    )

    x, x_path = lp_problem_2d.solve(mode='bruteforce')
    return

    visualize_func_in_2d_area(lp_problem_2d.objective_func(),
                              lp_problem_2d.optimization_area_indicator(),
                              title='x_plus_y_common')

    dual_problem = lp_problem_2d.from_common_to_dual()
    dual_dual_problem = dual_problem.from_common_to_dual()
    visualize_func_in_2d_area(dual_dual_problem.objective_func(),
                              dual_dual_problem.optimization_area_indicator(),
                              title='x_plus_y_dual_dual')

def danil_main():
    values = Interface().get_data()
    res = data_parser().get_output_data(values)
    if res is not None:
        x_dim, A, b_list, c_objective, M1_b_ineq, N1_x_positive = result
    else:
        #TODO: add message box
        return 0

if __name__ == '__main__':

    #viktor_test()
    mikhail_main()

    print('first lab')
