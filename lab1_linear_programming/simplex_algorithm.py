import numpy as np
import copy
import sys

TOLERANCE_POWER = 15

class SimplexAlgorithm:

    def __init__(self, A, b, c, ignore_phase1=False):
        self.tableau = None
        self.__construct_simple_tableau(A, b, c)

        self.basic_variables = []
        self.non_basic_variables = []
        idx_I_rows_exist = self.__find_basic_variables(do_division=True)

        self.no_solution = False

        if not ignore_phase1 and not self.__is_canonical_tableau():
            self.__phase1(idx_I_rows_exist)

    def __round_small_values(self):
        self.tableau = self.tableau.round(TOLERANCE_POWER)

    def __construct_simple_tableau(self, A, b, c):
        A, b, c = copy.deepcopy(A), copy.deepcopy(b), copy.deepcopy(c)

        A[b < 0] *= -1
        b[b < 0] *= -1

        self.tableau = np.row_stack([
            -c,
            A
        ])
        first_column = np.zeros(self.tableau.shape[0])
        first_column[0] = 1.0
        last_column = np.insert(b, 0, 0.0)

        self.tableau = np.column_stack([
            first_column, self.tableau, last_column
        ])

    def __find_basic_variables(self, do_division=False):
        # columns which have only one non-zero element
        self.__round_small_values()
        col_idx_nonzero = []
        for col in range(1, self.tableau.shape[1] - 1):
            shit = self.tableau[:, col]
            nonzero_idx = list(self.tableau[:, col].nonzero()[0])
            if len(nonzero_idx) == 1 and \
                    self.tableau[nonzero_idx[0], col] * self.tableau[nonzero_idx[0], -1] >= 0:
                col_idx_nonzero.append(col)

        if len(col_idx_nonzero) == 0:
            self.basic_variables = []
            self.non_basic_variables = []
            return []

        tableau_nonzero_part = self.tableau[:, col_idx_nonzero]
        # indexes of non-zero elements in one_nonzero_columns
        row_idx_nonzero = list((tableau_nonzero_part != 0).argmax(axis=0))
        # values of non-zero elements
        values_nonzero = np.array(
            [self.tableau[row_idx_nonzero[i], col_idx_nonzero[i]] for i in range(len(col_idx_nonzero))])

        '''
        construct following table 'table_nonzero':
        column index of nonzero element:        ---col_idx_nonzero----
        row index of nonzero element in column: ---row_idx_nonzero---
        value of nonzero elements:              ---values_nonzero----
        '''
        table_nonzero = np.array([list(col_idx_nonzero),
                                  list(row_idx_nonzero),
                                  list(values_nonzero)])

        # drop all repeating columns of identity matrix from table_nonzero
        _, unique_idx_row = np.unique(table_nonzero[1, :], return_index=True)
        table_nonzero = table_nonzero[:, unique_idx_row]

        if do_division:
            for i in range(table_nonzero.shape[1]):
                self.tableau[int(table_nonzero[1, i]), :] /= table_nonzero[2, i]

        self.basic_variables = list(table_nonzero[0, :].astype(int))
        self.non_basic_variables = list(set(range(1, self.tableau.shape[1] - 1)) - set(self.basic_variables))

        return list((table_nonzero[1, :]).astype(int))

    def __is_canonical_tableau(self):
        A_num_rows = self.tableau.shape[0] - 1
        return len(self.basic_variables) == A_num_rows

    def __phase1(self, idx_I_rows_exist):
        num_A_rows = self.tableau.shape[0] - 1
        idx_I_rows_needed = list(set(range(1, self.tableau.shape[0])) - set(idx_I_rows_exist))
        I_pick_from = np.row_stack([
            np.zeros(num_A_rows),
            np.eye(num_A_rows)
        ])
        I_pick_from = np.column_stack([
            np.zeros(I_pick_from.shape[0]), I_pick_from
        ])
        '''
        I_pick_from has form:
        0 . . . 0
        . 1
        .   . 
        .     .
        0       1
        this way, if we need to have row with 1 on i-th position, we need
        to insert i-th column from matrix above
        '''
        I_part_to_insert = I_pick_from[:, idx_I_rows_needed]

        b_ph1 = self.tableau[:, -1]
        A_ph1 = np.column_stack([
            self.tableau[:, :-1], I_part_to_insert
        ])
        c_ph1 = np.sum(A_ph1[idx_I_rows_needed, :], axis=0)
        c_ph1[-len(idx_I_rows_needed):] = 0
        upper_b = np.sum(b_ph1[idx_I_rows_needed])

        phase1_simplex = SimplexAlgorithm(
            A=A_ph1,
            b=b_ph1,
            c=-c_ph1,
            ignore_phase1=True
        )
        phase1_simplex.tableau[0, -1] = upper_b

        phase1_simplex.__process_tableau()
        tableau_phase1 = phase1_simplex.tableau

        if tableau_phase1[0, -1] != 0:
            self.no_solution = True
            return

        b_new = tableau_phase1[1:, -1]
        tableau_new = tableau_phase1[1:, 1:-(len(idx_I_rows_needed) + 1)]
        tableau_new = np.column_stack([
            tableau_new, b_new
        ])
        self.tableau = copy.deepcopy(tableau_new)

    def __process_tableau(self, keep_x_path=False):
        self.__find_basic_variables(do_division=True)

        x_path = []
        idx_of_pivot_row, idx_of_pivot_column = self.__select_pivot_row_and_column()
        while idx_of_pivot_row is not None and idx_of_pivot_column is not None:
            if keep_x_path:
                pass
                x_path.append(self.__extract_solutions()[1])

            self.__change_basic_variables(idx_of_pivot_row, idx_of_pivot_column)
            self.__find_basic_variables(do_division=True)

            idx_of_pivot_row, idx_of_pivot_column = self.__select_pivot_row_and_column()

        x_path.append(self.__extract_solutions()[1])
        x_path.reverse()
        return x_path

    def __select_pivot_row_and_column(self):
        first_row = self.tableau[0, :]

        if (first_row[self.non_basic_variables] <= 0).all():
            return None, None

        idx_of_pivot_column = find_max(first_row[1:-1], np.array(self.non_basic_variables) - 1) + 1

        b = self.tableau[1:, -1]
        a = self.tableau[1:, idx_of_pivot_column]
        arr = b / a

        min_idx = find_min_non_negative(arr)
        if min_idx is None:
            return None, None

        idx_of_pivot_row = min_idx + 1

        return idx_of_pivot_row, idx_of_pivot_column

    def __change_basic_variables(self, idx_of_pivot_row, idx_of_pivot_column):
        pivot_value = self.tableau[idx_of_pivot_row][idx_of_pivot_column]

        for row_idx in range(self.tableau.shape[0]):
            if row_idx == idx_of_pivot_row:
                continue
            factor = self.tableau[row_idx, idx_of_pivot_column] / pivot_value
            self.tableau[row_idx] -= self.tableau[idx_of_pivot_row] * factor

    def solve(self):
        if self.no_solution:
            return None, []

        x_path = self.__process_tableau(keep_x_path=True)
        if len(x_path) == 0:
            return None, []

        return x_path[0], x_path

    def __extract_solutions(self):
        solutions = np.zeros(len(self.basic_variables) + len(self.non_basic_variables))
        for i in self.basic_variables:
            row_idx = np.nonzero(self.tableau[:, i])[0][0]
            value_b = self.tableau[row_idx, -1]
            value_a = self.tableau[row_idx, i]
            solution_value = value_b / value_a

            solutions[i - 1] = 0 if solution_value < 0 else solution_value

        minimum_value = self.tableau[0, -1] / self.tableau[0, 0]
        return minimum_value, solutions


def find_max(array, indexes):
    max_value = 1
    first_positive_elem_idx = next(x for x, val in enumerate(array) if val > 0)
    max_index = first_positive_elem_idx
    for i in indexes:
        if array[i] > max_value:
            max_value = array[i]
            max_index = i
    return max_index


def find_min_non_negative(arr):
    min_value = sys.float_info.max
    min_idx = 0
    for i in range(len(arr)):
        if 0 < arr[i] < min_value:
            min_value = arr[i]
            min_idx = i
    return None if min_value == sys.float_info.max else min_idx