import numpy as np
from typing import Dict, List
from copy import copy
from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'])


class CitiesMap:

    DUMMY_CITY_NAME = 'V*'
    NAN_VALUE = np.inf

    def __init__(self, cities_coords: Dict[str, Point],
                 dict_path_cities: Dict[str, Dict[str, float]], ports_names: List[str],
                 river_profit_scale):

        self.__ports_names = copy(ports_names)
        self.__cities_names = list(cities_coords.keys()) + [CitiesMap.DUMMY_CITY_NAME]
        self.__cities_coords = copy(cities_coords)
        self.__cities_coords[CitiesMap.DUMMY_CITY_NAME] = (CitiesMap.NAN_VALUE, CitiesMap.NAN_VALUE)

        self.__cities_names_to_idx = {name: idx for idx, name in enumerate(self.__cities_names)}

        self.__adj_matrix = np.full([len(self.__cities_names_to_idx), len(self.__cities_names_to_idx)],
                                    fill_value=CitiesMap.NAN_VALUE)
        self.__paths_by_river = set()  # Set[Tuple[str, str]]

        self.__river_profit_scale = river_profit_scale

        self.__add_paths_to_adj_matrix(dict_path_cities)
        self.__add_paths_from_ports_by_river()

        # route
        self.__route_city_start = None
        self.__route_city_end = None

    @property
    def adjacency_matrix(self):
        if not self.is_route_prepared:
            return self.__adj_matrix[:-1, :-1]

        return copy(self.__adj_matrix)

    def get_path(self, city_start, city_end):
        path_weight = self.__adj_matrix[self.__cities_names_to_idx[city_start], self.__cities_names_to_idx[city_end]]

        if path_weight == CitiesMap.NAN_VALUE:
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
        self.__route_city_start = city_start
        self.__route_city_end = city_end

        self.__add_paths_to_adj_matrix(
            {
                city_end: {CitiesMap.DUMMY_CITY_NAME: 0},
                CitiesMap.DUMMY_CITY_NAME: {city_start: 0}
            }
        )

    @property
    def is_route_prepared(self):
        return self.__route_city_start is not None and self.__route_city_end is not None

    def kill_route(self):
        self.__add_paths_to_adj_matrix(
            {
                self.__route_city_end: {CitiesMap.DUMMY_CITY_NAME: CitiesMap.NAN_VALUE},
                CitiesMap.DUMMY_CITY_NAME: {self.__route_city_start: CitiesMap.NAN_VALUE}
            }
        )
        self.__route_city_start = None
        self.__route_city_end = None

    def __add_paths_to_adj_matrix(self, dict_path_cities: Dict[str, Dict[str, float]]):
        for city_start, paths in dict_path_cities.items():
            for city_end, weight in paths.items():
                self.__adj_matrix[self.__cities_names_to_idx[city_start],
                                  self.__cities_names_to_idx[city_end]] = weight

    def __add_paths_from_ports_by_river(self):
        for port_name in self.__ports_names:
            for city_name in set(self.__cities_names) - {port_name, CitiesMap.DUMMY_CITY_NAME}:
                path_by_river, _ = self.__river_path_port_to_city(port_name, city_name)

                path_by_road = self.__adj_matrix[self.__cities_names_to_idx[port_name],
                                                 self.__cities_names_to_idx[city_name]]
                if path_by_river < path_by_road:
                    self.__paths_by_river.add((port_name, city_name))

                    self.__adj_matrix[self.__cities_names_to_idx[port_name],
                                      self.__cities_names_to_idx[city_name]] = min(path_by_river, path_by_road)

    def __river_path_port_to_city(self, port_name, city_name):
        if city_name == CitiesMap.DUMMY_CITY_NAME or port_name == CitiesMap.DUMMY_CITY_NAME:
            return None, None

        port_coord = Point(*self.__cities_coords[port_name])
        city_coord = Point(*self.__cities_coords[city_name])

        N = self.__river_profit_scale
        L = abs(port_coord.x - city_coord.x)
        B = abs(port_coord.y - city_coord.y)

        Y = B / np.sqrt(N**2 - 1)
        point_on_river = Point(port_coord.x + np.sign(city_coord.x - port_coord.x) * (L - Y), port_coord.y)
        if Y > L:
            Y = L
            point_on_river = None

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

    cities_map = CitiesMap(cities, cities_paths, ports, river_profit)
    cities_map.prepare_for_route('A', 'E')
    adj_matrix = cities_map.adjacency_matrix

    path_from_D_to_E = cities_map.get_path('C', 'E')
    bp = 0

if __name__ == '__main__':
    __main()
