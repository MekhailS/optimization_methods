import math
import numpy as np
from scipy.optimize import linprog


def main():
    with open("output/matrix.txt", "r") as matrix_file:
        matrix = [[-int(num) for num in line.split()] for line in matrix_file]

    with open("output/free_vector.txt", "r") as free_vec_file:
        free_vec = [[-int(num) * 2 for num in line.split()] for line in free_vec_file]

    with open("output/target_func.txt", "r") as target_func_file:
        target_func = [[float(num) for num in line.split()] for line in target_func_file]

    solution = linprog(c=target_func, A_ub=matrix, b_ub=free_vec, method='simplex').x



    not_zero = list()
    sum = 0
    for idx, num in enumerate(solution):
        if num != 0:
            not_zero.append([idx, math.ceil(num)])
            sum += math.ceil(num)
    print(not_zero)
    print(sum)

    matrix = np.array(matrix)
    matrix = matrix.T

    for i in not_zero:
        idx, num = i
        print(idx)
        print(matrix[idx])

    print(matrix.shape)


# main()

"""Simple travelling salesman problem between cities."""
#
# import numpy as np
# from ortools.constraint_solver import routing_enums_pb2
# from ortools.constraint_solver import pywrapcp
# from scipy.sparse.csgraph import floyd_warshall, csgraph_from_dense
#
#
# def create_data_model():
#     """Stores the data for the problem."""
#     G2_data = np.array([[0, 0, np.inf, np.inf, np.inf],
#                         [np.inf, 0, np.inf, 4, 3],
#                         [0, np.inf, 0, np.inf, np.inf],
#                         [np.inf, 4, 7, 0, 2],
#                         [np.inf, 3, 5, 2, 0]])
#
#     # G2_data = np.array([[np.inf, np.inf, 4, 3],
#     #                     [0, np.inf, np.inf, np.inf],
#     #                     [4, 7, np.inf, 2],
#     #                     [3, 5, 2, np.inf],
#     #                     ])
#
#     G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
#     dist_matrix, predecessors = floyd_warshall(csgraph=G2_sparse, directed=True, return_predecessors=True)
#
#     print(dist_matrix)
#
#     data = {}
#     # data['distance_matrix'] = [
#     #     [np.inf, 7, 4, np.inf],
#     #     [7, np.inf, 7, 5],
#     #     [4, 7, np.inf, 2],
#     #     [3, np.inf, np.inf, np.inf],
#     # ]  # yapf: disable
#
#     data['distance_matrix'] = dist_matrix
#
#     data['num_vehicles'] = 1
#     data['depot'] = 2
#     return data
#
#
# def print_solution(manager, routing, solution):
#     """Prints solution on console."""
#     print('Objective: {} miles'.format(solution.ObjectiveValue()))
#     index = routing.Start(0)
#     plan_output = 'Route for vehicle 0:\n'
#     route_distance = 0
#     while not routing.IsEnd(index):
#         plan_output += ' {} ->'.format(manager.IndexToNode(index))
#         previous_index = index
#         index = solution.Value(routing.NextVar(index))
#         route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
#     plan_output += ' {}\n'.format(manager.IndexToNode(index))
#     print(plan_output)
#     plan_output += 'Route distance: {}miles\n'.format(route_distance)
#
#
# def main():
#     """Entry point of the program."""
#     # Instantiate the data problem.
#     data = create_data_model()
#
#     # Create the routing index manager.
#     manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
#                                            data['num_vehicles'], data['depot'])
#
#     # Create Routing Model.
#     routing = pywrapcp.RoutingModel(manager)
#
#     def distance_callback(from_index, to_index):
#         """Returns the distance between the two nodes."""
#         # Convert from routing variable Index to distance matrix NodeIndex.
#         from_node = manager.IndexToNode(from_index)
#         to_node = manager.IndexToNode(to_index)
#         return data['distance_matrix'][from_node][to_node]
#
#     transit_callback_index = routing.RegisterTransitCallback(distance_callback)
#
#     # Define cost of each arc.
#     routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
#
#     # Setting first solution heuristic.
#     search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#     search_parameters.first_solution_strategy = (
#         routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
#
#     # Solve the problem.
#     solution = routing.SolveWithParameters(search_parameters)
#
#     # Print solution on console.
#     if solution:
#         print_solution(manager, routing, solution)


if __name__ == '__main__':
    #
    # print(dist_matrix)
    #
    # # print(G2_sparse)
    main()
