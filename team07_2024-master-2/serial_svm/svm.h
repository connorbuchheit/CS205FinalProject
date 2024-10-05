
#ifndef SVM_H
#define SVM_H

// #include "kernel.h"
#include <vector>
// // kernels
// #include "linear_kernel.h"
// #include "rbf_kernel.h"
// #include "polynomial_kernel.h"



#define MAX_FILE_NAME_SIZE 1024


class Matrix {
public:
    Matrix(int rows, int cols);
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    int getRows() const;
    int getCols() const;

private:
    int rows, cols;
    std::vector<double> data;
};


/**
 * @brief svm parameters needed by model
 * 
 */
struct svm_param{

    int kernel_type;
    double degree; // for polynomial kernel
    double gamma; // for rbf 
};


/**
 * @brief saves data read from input file 
 * 
 */
struct problem {



};


struct serial_svm_model {

    struct svm_param param; // parameters

};

#endif // SVM_H
