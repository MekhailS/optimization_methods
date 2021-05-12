import numpy as np
from typing import Dict, List
from copy import deepcopy, copy
from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'])


class Map:

    DUMMY_CITY_NAME = 'V*'
    NAN_VALUE = np.inf

    def __init__(self, cities_coords: Dict[str, Point],
                 dict_path_cities: Dict[str, Dict[str, int]], ports_names: List[str],
                 river_profit_scale):

        self.__ports_names = copy(ports_names)
        self.__cities_names = list(cities_coords.keys()) + [Map.DUMMY_CITY_NAME]
        self.__cities_coords = deepcopy(cities_coords)
        self.__cities_coords[Map.DUMMY_CITY_NAME] = (Map.NAN_VALUE, Map.NAN_VALUE)

        self.__cities_names_to_idx = {name: idx for idx, name in enumerate(self.__cities_names)}

        self.__adj_matrix = np.full([len(self.__cities_names_to_idx), len(self.__cities_names_to_idx)],
                                    fill_value=Map.NAN_VALUE)
        self.__paths_by_river = set()  # Set[Tuple[str, str]]

        self.__river_profit_scale = river_profit_scale

        self.__add_paths_to_adj_matrix(dict_path_cities)
        self.__add_paths_from_ports_by_river()

        # route
        self.route_city_start = None
        self.route_city_end = None

    @property
    def adjacency_matrix(self):
        if not self.is_route_prepared:
            return self.__adj_matrix[:-1, :-1]

        return copy(self.__adj_matrix)

    def get_path(self, city_start, city_end):
        path_weight = self.__adj_matrix[self.__cities_names_to_idx[city_start], self.__cities_names_to_idx[city_end]]

        if path_weight == Map.NAN_VALUE:
            return [], None, None

        path_type = 'road'
        path = [(city_start, Point(*self.__cities_coords[city_start])),
                (city_end, Point(*self.__cities_coords[city_end]))]
        path_weight = self.__adj_matrix[self.__cities_names_to_idx[city_start],
                                        self.__cities_names_to_idx[city_end]]

        if (city_start, city_end) in self.__paths_by_river:
            path_type = 'river'
            _, point_on_river = self.__river_path_port_to_city(city_start, city_end)
            if point_on_river is not None:
                path.insert(1, ('RIVER', point_on_river))

        return path, path_weight, path_type

    def city_name_to_idx(self, city_name):
        return self.__cities_names_to_idx[city_name]

    def city_idx_to_name(self, city_idx):
        return self.__cities_names[city_idx]

    def prepare_for_route(self, city_start, city_end):
        self.route_city_start = city_start
        self.route_city_end = city_end

        self.__add_paths_to_adj_matrix(
            {
                city_end: {Map.DUMMY_CITY_NAME: 0},
                Map.DUMMY_CITY_NAME: {city_start: 0}
            }
        )

    @property
    def is_route_prepared(self):
        return self.route_city_start is not None and self.route_city_end is not None

    def kill_route(self):
        self.__add_paths_to_adj_matrix(
            {
                self.route_city_end: {Map.DUMMY_CITY_NAME: Map.NAN_VALUE},
                Map.DUMMY_CITY_NAME: {self.route_city_start: Map.NAN_VALUE}
            }
        )
        self.route_city_start = None
        self.route_city_end = None

    def __add_paths_to_adj_matrix(self, dict_path_cities: Dict[str, Dict[str, int]]):
        for city_start, paths in dict_path_cities.items():
            for city_end, weight in paths.items():
                self.__adj_matrix[self.__cities_names_to_idx[city_start],
                                  self.__cities_names_to_idx[city_end]] = weight

    def __add_paths_from_ports_by_river(self):
        for port_name in self.__ports_names:
            for city_name in set(self.__cities_names) - {port_name}:
                path_by_river, _ = self.__river_path_port_to_city(port_name, city_name)

                path_by_road = self.__adj_matrix[self.__cities_names_to_idx[port_name],
                                                 self.__cities_names_to_idx[city_name]]
                if path_by_river < path_by_road:
                    self.__paths_by_river.add((port_name, city_name))

                    self.__adj_matrix[self.__cities_names_to_idx[port_name],
                                      self.__cities_names_to_idx[city_name]] = min(path_by_river, path_by_road)

    def __river_path_port_to_city(self, port_name, city_name):
        port_coord = Point(*self.__cities_coords[port_name])
        city_coord = Point(*self.__cities_coords[city_name])

        N = self.__river_profit_scale
        L = abs(port_coord.x - city_coord.x)
        B = abs(port_coord.y - city_coord.y)

        point_on_river = None
        Y = B / np.sqrt(N**2 - 1)
        if Y > L:
            Y = L
        else:
            point_on_river = Point(port_coord.x + np.sign(city_coord.x - port_coord.x)*(L-Y), port_coord.y)

        path_weight = (L - Y) / N + np.sqrt(Y**2 + B**2)
        return path_weight, point_on_river


def __main():
    cities = {
        'A': (3, 7),
        'B': (1, 2),
        'C': (3, 0),
        'D': (6, 0),
        'E': (0, 6)
    }
    cities_paths = {
        'A': {'B': 8, 'C': 4, 'D': 3, 'E': 50},
        'B': {'A': 8, 'C': 1, 'E': 5},
        'C': {'A': 4, 'E': 7, 'B': 1},
        'D': {'A': 3, 'E': 5},
        'E': {}
    }
    ports = ['C', 'D']
    river_profit = 10

    map = Map(cities, cities_paths, ports, river_profit)
    map.prepare_for_route('A', 'E')
    adj_matrix = map.adjacency_matrix
    bp = 0

if __name__ == '__main__':
    __main()
