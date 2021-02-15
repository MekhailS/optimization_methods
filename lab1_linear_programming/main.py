
from lab1_linear_programming.data_parser import *
from lab1_linear_programming.interface import *
from lab1_linear_programming.plot_function import visualize_func_in_2d_area
from lab1_linear_programming.linear_programming import *
from lab1_linear_programming.interface import *

def viktor_main():
    # lp_problem_2d = LPProblem(
    #     x_dim=2,
    #     A=[[-1.0, -2.0],
    #        [-1.0, -1.0],
    #        [-3.0, -2.0]],
    #     b=[-16.0, -9.0, -24.0],
    #     c_objective=[-40.0, -30.0],
    #     M1_b_ineq=[0, 1, 2],
    #     N1_x_positive=[0, 1]
    # )
    # lp_problem_2d = LPProblem(
    #     x_dim=3,
    #     A=[[-1.0, -1.0, 4.0],
    #        [-1.0, 1.0, -7.0],
    #        [2.0, 1.0, 3.0]],
    #     b=[20.0, -20.0, 1.0],
    #     c_objective=[1.0, 1.0, -7.0],
    #     M1_b_ineq=[0, 1],
    #     N1_x_positive=[0, 1, 2]
    # )
    # lp_problem_2d = LPProblem(
    #     x_dim=2,
    #     A=[[-2.0, -1.0],
    #        [-6.0, -5.0],
    #        [-2.0, -5.0]],
    #     b=[-18.0, -60.0, -40.0],
    #     c_objective=[-2.0, -3.0],
    #     M1_b_ineq=[0, 1, 2],
    #     N1_x_positive=[0, 1]
    # )
    lp_problem_2d = LPProblem(
        x_dim=2,
        A=[[-1.0, -1.0],
           [1.0, -3.0],
           [1.0, -1.0]],
        b=[-3.0, -1.0, -3.0],
        c_objective=[-1.0, -1.0],
        M1_b_ineq=[0, 1, 2],
        N1_x_positive=[0, 1]
    )
    lp_problem_2d = LPProblem(
        x_dim=3,
        A=[[-1.0, -3.0, -2.0],
           [-1.0, -5.0, -1.0],
           [4.0, -3.0, -1.0]],
        b=[1.0, 1.0, -2.0],
        c_objective=[-8.0, -10.0, 3.0],
        M1_b_ineq=[0, 1, 2],
        N1_x_positive=[0, 1, 2]
    )
    lp_problem_2d = LPProblem(
        x_dim=2,
        A=[[-1.0, -2.0],
           [1.0, 1.0],
           [1.0, 3.0]],
        b=[-10.0, -20.0, 1],
        c_objective=[1.0, 1.0],
        M1_b_ineq=[0, 1, 2],
        N1_x_positive=None
    )
    lp_problem_test = lp_problem_2d.canonical()
    simplex = SimplexAlgorithm(
        A=lp_problem_test.A,
        b=lp_problem_test.b,
        c=lp_problem_test.c_objective
    )
    simplex.solve()
    lp_problem_2d.solve(mode='simplex')

    print("f")


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

# def danil_main():
#     values = Interface().get_data()
#     res = data_parser().get_output_data(values)
#     if res is not None:
#         x_dim, A, b_list, c_objective, M1_b_ineq, N1_x_positive = result
#     else:
#         #TODO: add message box
#         return 0

if __name__ == '__main__':

    viktor_main()
    # mikhail_main()

    print('first lab')
