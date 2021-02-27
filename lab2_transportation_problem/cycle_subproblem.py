import numpy as np
from enum import Enum


class Direction(Enum):
    HORIZONTAL = 1
    VERTICAL = 2


class VertexDFS:

    def __init__(self, i, j, direction):
        self.i = i
        self.j = j
        self.direction = direction


class CycleSubproblem:

    def __init__(self, A_matrix, val_empty=np.nan):
        """
        Create cycle subproblem by given matrix from transportation problem
        and values associated as empty

        :param A_matrix: cost matrix from transportation problem (after n-w method)
        :param val_empty: values in matrix to be associated as empty
        """
        A_matrix = np.array(A_matrix)

        # create table with given matrix
        # 0— empty values (val_empty)
        # 1 — cell wasn't visited
        # 2 — cell was visited
        self.__table = np.full(A_matrix.shape, 0, dtype=int)
        self.__table[A_matrix != val_empty] = 1

        self.__idx_nonempty_in_cols = []
        self.__idx_nonempty_in_rows = []

        for i in range(self.__table.shape[1]):
            self.__idx_nonempty_in_cols.append(np.sort(np.where(self.__table[:, i] != 0)[0]))

        for i in range(self.__table.shape[0]):
            self.__idx_nonempty_in_rows.append(np.sort(np.where(self.__table[i, :] != 0)[0]))

    def __mark_cell_as_visited(self, vertex_dfs):
        self.__table[vertex_dfs.i, vertex_dfs.j] = 2

    def __is_cell_visited(self, vertex_dfs):
        return self.__table[vertex_dfs.i, vertex_dfs.j] == 2

    def __adjacent_vertices(self, vertex_dfs):
        adjacent_vertices = []

        if vertex_dfs.direction == Direction.VERTICAL:
            for i in self.__idx_nonempty_in_cols[vertex_dfs.j]:
                if i != vertex_dfs.i:
                    adjacent_vertices.append(VertexDFS(i, vertex_dfs.j, Direction.HORIZONTAL))

        elif vertex_dfs.direction == Direction.HORIZONTAL:
            for j in self.__idx_nonempty_in_rows[vertex_dfs.i]:
                if j != vertex_dfs.j:
                    adjacent_vertices.append(VertexDFS(vertex_dfs.i, j, Direction.VERTICAL))

        return adjacent_vertices

    def find_cycle(self, start_i, start_j):
        """
        Find closed cycle in table of transportation problem
        (path with start and end at (start_i, start_j)

        :param start_i: row index in cost matrix of start element
        :param start_j: column index in cost matrix of start element
        :return: path as list of [i, j] — (row, column) indices of elements
        which form a closed cycle (starting element included)
        if there is no path, empty list is returned
        """
        target = None
        previous_vertex_map = {}

        stack_vertex_dfs = []
        stack_vertex_dfs.append(VertexDFS(start_i, start_j, Direction.VERTICAL))

        while stack_vertex_dfs:
            vertex_dfs_cur = stack_vertex_dfs[-1]
            stack_vertex_dfs.pop()
            if self.__is_cell_visited(vertex_dfs_cur):
                continue

            self.__mark_cell_as_visited(vertex_dfs_cur)
            if vertex_dfs_cur.i == start_i and vertex_dfs_cur.j != start_j:
                target = vertex_dfs_cur
                break

            adjacent_vertices = self.__adjacent_vertices(vertex_dfs_cur)
            for vertex in adjacent_vertices:
                stack_vertex_dfs.append(vertex)
                previous_vertex_map[(vertex.i, vertex.j)] = (vertex_dfs_cur.i, vertex_dfs_cur.j)

        if target is None:
            return []

        path = [[target.i, target.j]]
        cell_curr = previous_vertex_map[target.i, target.j]
        while True:
            path.append([cell_curr[0], cell_curr[1]])

            if cell_curr not in previous_vertex_map:
                break
            cell_curr = previous_vertex_map[cell_curr]

        return list(reversed(path))

