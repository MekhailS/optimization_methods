import numpy as np
from typing import Dict, List
from copy import deepcopy, copy
from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'])


class Map:

    NAN_VALUE = np.inf

    def __init__(self, cities_coords: Dict[str, Point],
                 dict_path_cities: Dict[str, Dict[str, int]], ports_names: List[str], river_profit_scale: float):

        self.ports_names = copy(ports_names)
        self.cities_names = deepcopy(cities_coords.keys())
        self.cities_coords = deepcopy(cities_coords)

        self.river_profit_scale = river_profit_scale
        self.cities_names_to_idx = dict(zip(self.cities_names, list(range(len(cities_coords)))))
        self.adj_matrix = np.full([len(self.cities_names_to_idx), len(self.cities_names_to_idx)],
                                  fill_value=Map.NAN_VALUE)
        self.paths_by_river = set()  # Set[Tuple[str, str]]

        self.__build_adj_matrix_from_paths(dict_path_cities)
        self.__add_paths_from_ports_by_river()

    @property
    def adjacency_matrix(self):
        return deepcopy(self.adj_matrix)

    def get_path(self, city_start, city_end):
        path_weight = self.adj_matrix[self.cities_names_to_idx[city_start], self.cities_names_to_idx[city_end]]

        if path_weight == Map.NAN_VALUE:
            return [], None, None

        path_type = 'road'
        path = [(city_start, self.cities_coords[city_start]), (city_end, self.cities_coords[city_end])]
        path_weight = self.adj_matrix[self.cities_names_to_idx[city_start],
                                      self.cities_names_to_idx[city_end]]

        if (city_start, city_end) in self.paths_by_river:
            path_type = 'river'
            _, point_on_river = self.__river_path_port_to_city(city_start, city_end)
            if point_on_river is not None:
                path.insert(1, ('RIVER', point_on_river))

        return path, path_weight, path_type

    def city_name_to_index(self, city_name):
        return self.cities_names_to_idx[city_name]

    def __build_adj_matrix_from_paths(self, dict_path_cities: Dict[str, Dict[str, int]]):
        for city_start, paths in dict_path_cities.items():
            for city_end, weight in paths.values():
                self.adj_matrix[self.cities_names_to_idx[city_start],
                                self.cities_names_to_idx[city_end]] = weight

    def __add_paths_from_ports_by_river(self):
        for port_name in self.ports_names:
            for city_name in set(self.cities_names) - {port_name}:
                path_by_river, _ = self.__river_path_port_to_city(port_name, city_name)

                path_by_road = self.adj_matrix[self.cities_names_to_idx[port_name],
                                               self.cities_names_to_idx[city_name]]
                if path_by_river < path_by_road:
                    self.paths_by_river.add((port_name, city_name))

                    self.adj_matrix[self.cities_names_to_idx[port_name],
                                    self.cities_names_to_idx[city_name]] = min(path_by_river, path_by_road)

    def __river_path_port_to_city(self, port_name, city_name):
        port_coord = self.cities_coords[port_name]
        city_coord = self.cities_coords[city_name]

        N = self.river_profit_scale
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
