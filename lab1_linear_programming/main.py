from lab1_linear_programming.data_parser import *
from lab1_linear_programming.interface import *
from lab1_linear_programming.plot_function import *
from lab1_linear_programming.linear_programming import *

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
        x_dim=3,
        A=[[-1.0, -1.0, 4.0],
           [-1.0, 1.0, -7.0],
           [2.0, 1.0, 3.0]],
        b=[20.0, -20.0, 1.0],
        c_objective=[1.0, 1.0, -7.0],
        M1_b_ineq=None,
        N1_x_positive=[0]
    )
    lp_problem_2d = LPProblem(
        x_dim=2,
        A=[[-1.0, -2.0],
           [1.0, 1.0],
           [1.0, 3.0]],
        b=[-10.0, -20.0, 1],
        c_objective=[-1.0, -4.0],
        M1_b_ineq=[0, 1, 2],
        N1_x_positive=[0, 1],
        obj_direction=ObjectiveDirection.MAX
    )
    lp_problem_2d = LPProblem(
        x_dim=2,
        A=[[-1, -4],
           [-4, -2],
           [-1, 5],
           [1, 14],
           [1, 2],
           [2, 1],
           [2, -3]],
        b=[-100, -170, -300, -700, -100, -100, -140],
        c_objective=[3, -1],
        M1_b_ineq=[0, 1, 2, 3, 4, 5, 6]
    )
    lp_problem_2d = LPProblem(
        x_dim=5,
        A=[[1, 4, 5, 6, 2],
           [3, -5, -1, -4, -1],
           [4, -3, -3, 0, -1],
           [7, 0, 10, 3, 4],
           [1, 4, 3, -8, 9]],
        b=[2, -10, 4, 3, 2],
        c_objective=[3, -4, 2, 1, 4],
        M1_b_ineq=[0, 1],
        N1_x_positive=[0, 1, 2, 4]
    )
    good_seeds = [5112]
    np.random.seed(212)
    lp_problem_2d = LPProblem(
        x_dim=5,
        A=(np.random.rand(5, 5)*10).round(0),
        b=(np.random.rand(5)*10).round(0),
        c_objective=[3, -4, 2, 1, 4],
        M1_b_ineq=[0, 1],
        N1_x_positive=[0, 1, 2, 4]
    )
    res_real = linprog(
        c=lp_problem_2d.c_objective,
        A_ub=-lp_problem_2d.A[lp_problem_2d.M1_b_ineq, :],
        b_ub=-lp_problem_2d.b[lp_problem_2d.M1_b_ineq],
        A_eq=-lp_problem_2d.A[lp_problem_2d.M2_b_eq, :],
        b_eq=-lp_problem_2d.b[lp_problem_2d.M2_b_eq],
        bounds=[(0, None), (0, None), (0, None), (None, None), (0, None)],
    )
    res_canon, _ = lp_problem_2d.solve(mode='scipy')

    lp_problem_dual = lp_problem_2d.dual()
    res_real_dual = linprog(
        c=lp_problem_dual.c_objective,
        A_ub=-lp_problem_dual.A[lp_problem_2d.M1_b_ineq, :],
        b_ub=-lp_problem_dual.b[lp_problem_2d.M1_b_ineq],
        A_eq=-lp_problem_dual.A[lp_problem_2d.M2_b_eq, :],
        b_eq=-lp_problem_dual.b[lp_problem_2d.M2_b_eq],
        bounds=[(0, None), (0, None), (None, None), (None, None), (None, None)],
    )
    res_canon_dual, _ = lp_problem_dual.solve(mode='scipy')
    #lp_problem_2d.dual(inplace=True)
    res_scipy, _ = lp_problem_2d.solve(mode='scipy')
    res_bruteforce, path_bruteforce = lp_problem_2d.solve(mode='bruteforce')
    res_simplex, path_simplex = lp_problem_2d.solve(mode='simplex')

    #val1 = np.array([3, -4, 2, 1, 4]) @ res_simplex
    #val2 = np.array([3, -4, 2, 1, 4]) @ res_bruteforce
    res = [lp_problem_2d.c_objective @ el for el in path_bruteforce]
    #vec = lp_problem_2d.A @ res_simplex
    print("f")


def mikhail_main():

    lp_problem_2d = LPProblem(
        x_dim=2,
        A=[[-1, -4],
           [-4, -2],
           [-1, 5],
           [1, 14],
           [1, 2],
           [2, 1],
           [2, -3]],
        b=[-100, -170, -300, -700, -100, -100, -140],
        c_objective=[3, -1],
        M1_b_ineq=[0, 1, 2, 3, 4, 5, 6],
        obj_direction=ObjectiveDirection.MAX
    )
    res_scipy, _ = lp_problem_2d.solve(mode='scipy')
    res_bruteforce, path_bruteforce = lp_problem_2d.solve(mode='bruteforce')
    res_simplex, path_simplex = lp_problem_2d.solve(mode='simplex')

    fig, ax = visualize_func_in_2d_area(lp_problem_2d.objective_func(),
                              lp_problem_2d.optimization_area_indicator(),
                              title='x_plus_y_common')
    draw_points(fig, ax, path_bruteforce, title='bruteforce solution', style_points='yo')
    draw_points(fig, ax, path_simplex, title='simplex algorithm solution', style_points='mx',
                use_arrow=True, color_arrow='magenta')
    plt.show()


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
