import numpy as np
import copy
from lab1_linear_programming.plot_function import visualize_func_in_2d_area
from lab1_linear_programming.linear_programming import *


if __name__ == '__main__':
    lp_problem = LPProblem([[5, 2, 3, 6, -2],
                                [10, 5, 6, 43, 10],
                                [6, -7, -2, 1, 4],
                                [1, 3, 4, 6, 8],
                                [4, 7, 3, 2, 1]],
                           [5, 10, 1, 1, 0],
                           [2, 3, 4, 1, 1],
                           [3 ,4], [0, 1, 2], [1 ,3], [0, 2, 4]
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

    # lp_problem_2d = LPProblem(x_size=2,
    #                           LS_eq=None,
    #                           LS_ineq=LinearSystem(
    #                               [[-1, -1],
    #                                [-1, 1],
    #                           ),
    #                                [2, 1]],
    #                               [-20, -20, -1],
    #                               relationship='>='
    #                           x_positive_indexes=[0, 1],
    #                           c_objective=[1, 1]
    #                           )
    # t = lp_problem_2d.dualLPProblem()
    # y = t.dualLPProblem()
    # visualize_func_in_2d_area(lp_problem_2d.objective_func(),
    #                           lp_problem_2d.optimization_area_indicator())
    print('first lab')
