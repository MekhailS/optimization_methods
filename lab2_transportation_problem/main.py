from transportation_problem import *

if __name__ == '__main__':
    A = [[0, 16, 3, 11, 12, 25],
         [22, 7, 1, 4, 14, 10],
         [17, 3, 7, 4, 9, 13],
         [11, 1, 6, 7, 8, 8],
         [12, 1, 6, 7, 8, 8]]

    transport_problem = TransportProblem(A)
    # if ~transport_problem.is_closed_problem():
    #     transport_problem.to_closed_problem()

