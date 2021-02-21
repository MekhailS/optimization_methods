from transportation_problem import *

if __name__ == '__main__':
    A = [[0, 16, 3, 11, 12, 25],
         [22, 7, 1, 4, 14, 10],
         [17, 3, 7, 4, 9, 13],
         [11, 1, 6, 7, 8, 8],
         [12, 1, 6, 7, 8, 8]]

    A_Rodionova = \
        [[0, 3, 6, 5, 6],
         [7, 3, 11, 3, 10],
         [4, 1, 9, 2, 8],
         [9, 7, 4, 10, 5]]
    transport_problem = TransportProblem(A_Rodionova)
    transport_problem.northwest_corner_method()
    print(transport_problem.obj_function_value())
    # if ~transport_problem.is_closed_problem():
    #     transport_problem.to_closed_problem()

