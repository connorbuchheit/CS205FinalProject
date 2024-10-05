// serial_icf.cpp

#include "serialicf.h"
#include <algorithm>
#include <iostream>

// Matrix class definitions
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, 0.0) {}

double& Matrix::operator()(int row, int col) {
    return data[row * cols + col];
}

const double& Matrix::operator()(int row, int col) const {
    return data[row * cols + col];
}

int Matrix::getRows() const { return rows; }
int Matrix::getCols() const { return cols; }

// serialICF function definition
void serialICF(const Matrix& Q, int n, int p, Matrix& H) {
    std::vector<double> v(n); // Assuming v is the diagonal of Q for simplicity.
    for (int i = 0; i < n; ++i) {
        v[i] = Q(i, i); // Assume Q(i, i) gives the diagonal elements.
    }

    for (int k = 0; k < p; ++k) {
        auto it = std::max_element(v.begin(), v.end());
        int ik = std::distance(v.begin(), it);
        double vpk = *it;
        H(ik, k) = std::sqrt(vpk);
        for (int j = 0; j < n; ++j) {
            if (j != ik) {
                double sum = 0.0;
                for (int l = 0; l < k; ++l) {
                    sum += H(j, l) * H(ik, l);
                }
                H(j, k) = (Q(j, ik) - sum) / H(ik, k);
            }
        }
        for (int j = 0; j < n; ++j) {
            v[j] -= H(j, k) * H(j, k);
        }
    }
}

// Function to calculate trace of Q - HH^T
double calculateTraceDifference(const Matrix& Q, const Matrix& H) {
    int n = Q.getRows();
    double trace = 0.0;
    Matrix HHt(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            HHt(i, j) = 0.0;
            for (int k = 0; k < H.getCols(); ++k) {
                HHt(i, j) += H(i, k) * H(j, k);
            }
        }
    }

    // Calculate trace of Q - HH^T
    for (int i = 0; i < n; ++i) {
        trace += Q(i, i) - HHt(i, i);
    }

    return trace;
}

// Function to check
// One question here is how to choose epsilon?
bool checkICFCriterion(const Matrix& Q, const Matrix& H, double epsilon) {
    double traceDiff = calculateTraceDifference(Q, H);

    if (traceDiff < epsilon) {
        std::cout << "Criterion met: tr(Q - HH^T) = " << traceDiff << " < " << epsilon << std::endl;
        return true;
    } else {
        std::cout << "Criterion not met: tr(Q - HH^T) = " << traceDiff << " >= " << epsilon << std::endl;
        return false;
    }
}