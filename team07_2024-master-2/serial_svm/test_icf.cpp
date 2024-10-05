// main.cpp

#include "serialicf.h"
#include <iostream>

// Helper function to print matrices

void printMatrix(const Matrix& mat) {
    for (int i = 0; i < mat.getRows(); ++i) {
        for (int j = 0; j < mat.getCols(); ++j) {
            std::cout << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
// can make with g++ -std=c++11 -Wall test_icf.cpp serialicf.cpp -o serialICFApp
int main() {
    // test case 1
    int n = 4;
    int p = 2;
    double epsilon = 1e-4;
    Matrix Q(n, n);
    Q(0, 0) = 4; Q(0, 1) = 1; Q(1, 0) = 1; Q(1, 1) = 3;
    Q(2, 2) = 4; Q(2, 3) = 1; Q(3, 2) = 1; Q(3, 3) = 3;
    Matrix H(n, p);
    serialICF(Q,4,2, H);
    checkICFCriterion(Q, H, epsilon);
    printMatrix(Q);
    Matrix HHt(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            HHt(i, j) = 0.0;
            for (int k = 0; k < H.getCols(); ++k) {
                HHt(i, j) += H(i, k) * H(j, k);
            }
        }
    }
    // take a look at HHt;
    printMatrix(HHt);
    return 0;
}