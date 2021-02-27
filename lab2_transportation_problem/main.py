from lab2_transportation_problem.transportation_problem import *
from lab2_transportation_problem.cycle_subproblem import CycleSubproblem


# def mikhail_main():
#     A_matrix = [
#         [0, 1, 3, 0, 1],
#         [0, 0, 0, 0, 1],
#         [3, 4, 0, 0, 2]
#     ]
#     cycle_subproblem = CycleSubproblem(A_matrix, val_empty=0)
#     path = cycle_subproblem.find_cycle(1, 1)
#     k = 0

if __name__ == '__main__':
    A = [[0, 16, 3, 11, 12, 20],
         [22, 7, 1, 4, 14, 10],
         [17, 3, 7, 4, 9, 13],
         [11, 1, 6, 7, 8, 8],
         [12, 3, 2, 5, 6, 7]]

    A_Rodionova = \
        [[0, 3, 6, 5, 6],
         [7, 3, 11, 3, 10],
         [4, 1, 9, 2, 8],
         [9, 7, 4, 10, 5]]

    A_bad = \
        [[0, 6, 3, 4, 8, 6],
         [6, 3, 4, 4, 3, 1],
         [11, 5, 6, 5, 3, 1],
         [5, 4, 3, 2, 6, 1],
         [5, 5, 8, 6, 4, 5]]

    transport_problem = TransportProblem(A)
    transport_problem.potential_method()
    # transport_problem.northwest_corner_method()
    # print(transport_problem.obj_function_value())
    # if ~transport_problem.is_closed_problem():
    #     transport_problem.to_closed_problem()

