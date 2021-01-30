import numpy as np
import copy
from lab1_linear_programming.plot_function import visualize_func_in_2d_area
from lab1_linear_programming.linear_programming import *


if __name__ == '__main__':
    lp_problem = LPProblem(5,
                           LinearSystem(
                               [[5, 2, 3, 6, -2],
                                [10, 5, 6, 43, 10],
                                [6, -7, -2, 1, 4]],
                               [5, 10, 1]
                           ),
                           LinearSystem(
                               [[1, 3, 4, 6, 8],
                                [4, 7, 3, 2, 1]],
                               [1, 0]
                           ),
                           [1, 3],
                           [2, 3, 4, 1, 1]
                           )
    lp_problem.canonical(inplace=True)

    lp_problem_2d = LPProblem(x_size=2,
                              LS_eq=None,
                              LS_ineq=LinearSystem(
                                  [[-1, -1],
                                   [-1, 1],
                                   [2, 1]],
                                  [-20, -20, -1]
                              ),
                              x_positive_indexes=[1],
                              c_objective=[1, 1]
                              )
    visualize_func_in_2d_area(lp_problem_2d.objective_func(),
                              lp_problem_2d.optimization_area_indicator())

    print('first lab')
