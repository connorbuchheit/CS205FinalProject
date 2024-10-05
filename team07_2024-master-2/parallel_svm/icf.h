// icf.h

#ifndef ICF_H
#define ICF_H
#include <eigen3/Eigen/Dense>
#include <omp.h>
#include <iostream>
#include <cstring>
#include <stdio.h>
#include <vector>
#include <mpi.h>
struct MaxLoc {
    double value;
    int index;
};
// void compute_kernel_matrix(Eigen::MatrixXd* Q, const std::vector<Eigen::VectorXd>& data, const Params& param);
MaxLoc find_global_max_index(const std::vector<double>& local_v, int local_size, int start_row, int rank, int size);
double parallelICF(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& localQ, int n, int p,Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& localH);
#endif // ICF_H