import numpy as np


class data_parser:
    def __init__(self):
        self.b_list = []
        self.A = []

    def __check_values(self, dictionary):
        self.x_dim = dict(dictionary).get(2)
        if self.x_dim is None or self.x_dim == '':
            return None
        else:
            self.x_dim = int(self.x_dim)
            return True

    def __separate_string(self, string, separator, expected_num_args=None):
        list_of_separations = str(string).split(separator)

        if list_of_separations[-1].isspace() or list_of_separations[-1] == '':
            del list_of_separations[-1]

        if expected_num_args is not None and len(list_of_separations) != expected_num_args:
            return None
        else:
            return list_of_separations

    def __separate_line(self, line, separator):
        list_of_separations = self.__separate_string(line, separator, 2)
        if list_of_separations is None:
            return None

        self.b_list.append(float(list_of_separations[1]))

        coefficients = self.__separate_string(list_of_separations[0], ' ', self.x_dim)

        coefficients = np.array(list(map(float, coefficients)))

        if separator == '<=':
            coefficients = coefficients * (-1)

        if coefficients is None:
            return None
        else:
            self.A.append(coefficients)
            return True

    def __parse_input_data(self, dictionary):
        dictionary = dict(dictionary)
        if self.__check_values(dictionary) is None:
            return None

        self.method = dictionary.get(0)

        self.c_objective = np.array(list(map(float, self.__separate_string(dictionary.get(1), ' ', self.x_dim + 1))))

        for i in range(3, 6):
            if self.__separate_line(dictionary.get(i), '=') is None:
                return None

        if self.__separate_line(dictionary.get(6), '<=') is None:
            return None

        if self.__separate_line(dictionary.get(7), '>=') is None:
            return None

        self.M1_b_ineq = [3., 4.]
        self.N1_x_positive = np.array(list(map(float, self.__separate_string(dictionary.get(8), ' '))))

        return True

    def get_output_data(self, dictionary):
        if self.__parse_input_data(dictionary) is None:
            return None
        else:
            return self.method, self.x_dim, self.A, np.array(self.b_list), self.c_objective, self.M1_b_ineq, self.N1_x_positive
