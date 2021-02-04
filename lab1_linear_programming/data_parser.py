import numpy as np

class data_parser:
    def __init__(self):
        self.b_list = []
        self.A = []

    def __check_values(self, dictionary):
        self.x_dim = dict(dictionary).get(1)
        if self.x_dim is None or self.x_dim == '':
            return None
        else:
            self.x_dim = int(self.x_dim)
            return True

    def __separate_string(self, string, separator, expected_num_args):
        list_of_separations = str(string).split(separator)

        if list_of_separations[-1].isspace() or list_of_separations[-1] == '':
            del list_of_separations[-1]

        if len(list_of_separations) != expected_num_args:
            return None
        else:
            return list_of_separations

    def __separate_line(self, line, separator):
        list_of_separations = self.__separate_string(line, separator, 2)
        if list_of_separations is None:
            return None

        self.b_list.append(list_of_separations[1])

        coefficients = self.__separate_string(list_of_separations[0], ' ', self.x_dim)
        if coefficients is None:
            return None
        else:
            self.A.append(np.array(coefficients))
            return True

    def __parse_input_data(self, dictionary):
        dictionary = dict(dictionary)
        if self.__check_values(dictionary) is None:
            return None

        self.c_objective = np.array(dictionary.get(0))

        for i in range(2, 5):
            if self.__separate_line(dictionary.get(i), '=') is None:
                return None

        if self.__separate_line(dictionary.get(5), '<=') is None:
            return None

        if self.__separate_line(dictionary.get(6), '>=') is None:
            return None

        self.M1_b_ineq = [3, 4]
        self.N1_x_positive = np.array(dictionary.get(7))

        return True

    def get_output_data(self, dictionary):
        if self.__parse_input_data(dictionary) is None:
            return None
        else:
            return self.x_dim, self.A, np.array(self.b_list), self.c_objective, self.M1_b_ineq, self.N1_x_positive



