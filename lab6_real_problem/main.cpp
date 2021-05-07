#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

int main() {
    std::ofstream matrix_file;
    std::ofstream free_vector_file;
    std::ofstream target_func_file;
    matrix_file.open("matrix.txt", std::ios::out);
    free_vector_file.open("free_vector.txt", std::ios::out);
    target_func_file.open("target_func.txt", std::ios::out);

    double L = 11.7;
    std::vector<double> lengths = {0.6, 0.68, 0.83, 1.61, 1.67, 1.79, 2.8, 3.25, 3.25, 3.7, 3.95};
    std::vector<int> amounts = {249, 60, 97, 76, 72, 18, 43, 5424, 450, 515, 28};
    std::vector<double> restrictions;
    for (const auto &leng : lengths)
        restrictions.push_back(ceil(L / leng));


    std::vector<std::vector<size_t>> idxs;
    std::vector<double> subs;

    for (size_t idx10 = 0; idx10 < restrictions[10]; ++idx10)
        for (size_t idx9 = 0; idx9 < restrictions[9]; ++idx9)
            for (size_t idx8 = 0; idx8 < restrictions[8]; ++idx8)
                for (size_t idx7 = 0; idx7 < restrictions[7]; ++idx7)
                    for (size_t idx6 = 0; idx6 < restrictions[6]; ++idx6)
                        for (size_t idx5 = 0; idx5 < restrictions[5]; ++idx5)
                            for (size_t idx4 = 0; idx4 < restrictions[4]; ++idx4)
                                for (size_t idx3 = 0; idx3 < restrictions[3]; ++idx3)
                                    for (size_t idx2 = 0; idx2 < restrictions[2]; ++idx2)
                                        for (size_t idx1 = 0; idx1 < restrictions[1]; ++idx1)
                                            for (size_t idx0 = 0; idx0 < restrictions[0]; ++idx0) {
                                                auto tmp = lengths[0] * idx0 + lengths[1] * idx1 + lengths[2] * idx2 +
                                                           lengths[3] * idx3 + lengths[4] * idx4 + lengths[5] * idx5 +
                                                           lengths[6] * idx6 + lengths[7] * idx7 + lengths[8] * idx8 +
                                                           lengths[9] * idx9 + lengths[10] * idx10;
                                                if (tmp <= L) {
                                                    idxs.push_back(
                                                            {idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9,
                                                             idx10});
                                                    subs.push_back(L - tmp);
                                                }
                                            }

    for (size_t column = 0; column < idxs[0].size(); ++column) {
        for (size_t row = 0; row < idxs.size(); ++row)
            matrix_file << idxs[row][column] << " ";
        matrix_file << std::endl;
    }

    for (const auto &el : amounts)
        free_vector_file << el << " ";

    for (const auto &el : subs)
        target_func_file << el << " ";

    matrix_file.close();
    free_vector_file.close();
    target_func_file.close();
    return 0;
}