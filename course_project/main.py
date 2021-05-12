import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.sparse.csgraph import floyd_warshall, csgraph_from_dense

from cities_map import CitiesMap


def path_w_names_and_cords(cities_map: CitiesMap, path):
    res_path = []

    for i in range(len(path)-1):
        path_part, _, _ = cities_map.get_path(
            cities_map.city_idx_to_name(path[i]),
            cities_map.city_idx_to_name(path[i+1])
        )
        if i > 0:
            path_part = path_part[1:]
        res_path += path_part

    str_path = ' -> '.join([str(city) for city in res_path])
    return str_path


def extract_path(matrix, from_idx, to_idx):
    res_path = []
    prev_idx = matrix[from_idx][to_idx]
    while prev_idx != -9999:
        res_path.insert(0, prev_idx)
        prev_idx = matrix[from_idx][prev_idx]
    res_path.append(to_idx)
    return res_path

def get_detailed_path(predecessors, manager, routing, solution):
    """Prints solution on console."""
    index = routing.Start(0)
    previous_index = index
    route_distance = 0
    detailed_path = []
    start = True
    while not routing.IsEnd(index):
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        res = extract_path(predecessors, manager.IndexToNode(previous_index), manager.IndexToNode(index))

        if not start:
            res = res[1:]

        detailed_path += res

        previous_index = index
        index = solution.Value(routing.NextVar(index))

        start = False

    res = extract_path(predecessors, manager.IndexToNode(previous_index), manager.IndexToNode(index))
    res = res[1:]
    detailed_path += res
    print(detailed_path)
    return detailed_path


def create_cities_map():
    cities = {
        'A': (3, 7),
        'B': (1, 2),
        'C': (3, 0),
        'D': (6, 0),
        'E': (9, 6),
        'F': (100, 100)
    }
    cities_paths = {
        'A': {'B': 8, 'C': 4, 'D': 3, 'E': 50},
        'B': {'A': 8, 'C': 10, 'E': 5},
        'C': {'A': 4, 'E': 7, 'B': 10},
        'D': {'A': 3, 'E': 5},
        'E': {'F': 10},
        'F': {'E': 10}
    }
    ports = ['C', 'D']
    river_profit = 10
    cities_map = CitiesMap(cities, cities_paths, ports, river_profit)

    route = 'A', 'E'
    cities_map.prepare_for_route(*route)

    return cities_map


def create_data_model():
    """Stores the data for the problem."""


    cities_map = create_cities_map()

    path_DB = cities_map.get_path('D', 'B')

    G2_data = np.array(
        [
            [np.inf, np.inf,      1, np.inf, np.inf],
            [np.inf, np.inf,      1,      2,      0],
            [np.inf,      1, np.inf, np.inf, np.inf],
            [np.inf,      2, np.inf, np.inf, np.inf],
            [     0, np.inf, np.inf, np.inf, np.inf]
        ]
    )
    G2_data = cities_map.adjacency_matrix

    G2_sparse = csgraph_from_dense(G2_data, null_value=CitiesMap.NAN_VALUE)
    dist_matrix, predecessors = floyd_warshall(csgraph=G2_sparse, directed=True, return_predecessors=True)

    data = {}

    data['distance_matrix'] = dist_matrix

    data['num_vehicles'] = 1
    data['depot'] = cities_map.city_name_to_idx(cities_map.route_city_end)
    return data, predecessors, cities_map


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} miles'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)


def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data, pred, cities_map = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    path_idx = get_detailed_path(pred, manager, routing, solution)
    print(path_idx)

    str_path = path_w_names_and_cords(cities_map, path_idx)
    print(str_path)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution)


if __name__ == '__main__':
    #
    # print(dist_matrix)
    #
    # # print(G2_sparse)
    main()
