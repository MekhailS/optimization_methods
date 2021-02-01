import numpy as np
import copy
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
        A=[[-1, -1],
           [-1, 1],
           [2, 1]],
        b=[-20, -20, -1],
        c_objective=[1, 1],
        M1_b_ineq=[0, 1, 2],
        N1_x_positive=None
    )

    canon = lp_problem_2d.canonical()
    return

    visualize_func_in_2d_area(lp_problem_2d.objective_func(),
                              lp_problem_2d.optimization_area_indicator(),
                              title='x_plus_y_common')

    dual_problem = lp_problem_2d.from_common_to_dual()
    dual_dual_problem = dual_problem.from_common_to_dual()
    visualize_func_in_2d_area(dual_dual_problem.objective_func(),
                              dual_dual_problem.optimization_area_indicator(),
                              title='x_plus_y_dual_dual')


if __name__ == '__main__':

    #viktor_test()
    mikhail_main()

    print('first lab')
