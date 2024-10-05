// icf.cpp

#include <eigen3/Eigen/Dense>
#include <omp.h>
// #include "kernel.h"
// #include "linear_kernel.h"
// #include "parallel_ipm.h"
// #include "rbf_kernel.h"
// #include "polynomial_kernel.h"
#include "icf.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <limits>
#include <cfloat>
// void compute_kernel_matrix(Eigen::MatrixXd* Q, const std::vector<Eigen::VectorXd>& data, const Params& param) {
//     int n_samples = data.size();
//     Q->resize(n_samples, n_samples);

//     LinearKernel linear_kernel;
//     PolynomialKernel poly_kernel;
//     RBFKernel rbf_kernel;
//     // Compute the Q matrix using the specified kernel function
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < n_samples; ++i) {
//         for (int j = 0; j < n_samples; ++j) {
//             switch(param.kernel_type){
//                 case 0:  // Linear kernel
//                     (*Q)(i, j) = linear_kernel(data[i], data[j], param);
//                     break;
//                 case 1:  // Polynomial kernel
//                     (*Q)(i, j) = poly_kernel(data[i], data[j], param);
//                     break;
//                 case 2:  // RBF kernel
//                     (*Q)(i, j) = rbf_kernel(data[i], data[j], param);
//                     break;
//                 default:  // Default case should also handle errors or use a default kernel
//                     (*Q)(i, j) = poly_kernel(data[i], data[j], param);
//                     break;
//             }
//         }
//     }
// }

double parallelICF(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& localQ, int n, int p, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& localH) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = n / size;
    int extra_rows = n % size;
    int rows_for_this_process = rows_per_process + (rank < extra_rows ? 1 : 0);
    int start_row = 0;
    for (int i = 0; i < rank; ++i) {
        start_row += rows_per_process + (i < extra_rows ? 1 : 0);
    }
    if (rank == 1){
        // std::cout<<"start row: "<<start_row<<std::endl;
    }
    int end_row = start_row + rows_for_this_process;
    std::vector<double> local_v(rows_for_this_process);
    for (int i = 0; i < rows_for_this_process; ++i) {
        // std::cout<<localQ(i,i)<<std::endl;
        local_v[i] = localQ(i, start_row+i);
        // std::cout<<local_v[i]<<std::endl;
    }
    // std::cout<<"rank: "<<rank<< "localv0: " <<local_v[0]<<std::endl;
    // std::cout<<"row for this process"<<rows_for_this_process<<std::endl;
    if (size == 1){
        // std::cout<<"rank: "<<rank<< "localv400: " <<local_v[400]<<std::endl;
    }
    std::vector<double> pivot_row(p);
    for (int k = 0; k < p; ++k) {
        double local_trace = 0;
        for (int i = 0; i < rows_for_this_process; i++) {
            local_trace += local_v[i];
        }
        double global_trace = DBL_MAX;
        // std::cout<<"local trace!"<<local_trace<<std::endl;
        MPI_Allreduce(&local_trace, &global_trace, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // new change here: threshold to determine stop
        // std::cout<<"global trace track"<<global_trace<<std::endl;
        if (global_trace < 5){
            return k;
        }
        MaxLoc global_max = find_global_max_index(local_v, rows_for_this_process, start_row, rank, size);
        int global_ik = global_max.index;
        double global_vpk = global_max.value;

        int master_rank = (global_ik >= start_row && global_ik < end_row) ? rank : -1;
        int final_master_rank;
        MPI_Allreduce(&master_rank, &final_master_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (rank == final_master_rank) { 
            // std::cout<<"pivot role val"<<std::endl;
            for (int l = 0; l <= k; ++l) {
                pivot_row[l] = (l < k) ? localH(global_ik - start_row, l) : std::sqrt(global_vpk);
                // std::cout<<pivot_row[l]<<std::endl;
            }
        }
        MPI_Bcast(pivot_row.data(), k + 1, MPI_DOUBLE, final_master_rank, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int j = 0; j < rows_for_this_process; ++j) {
            // std::cout<<"j, k"<< j << k << std::endl;
            if ((start_row + j) != global_ik) {
                double sum = 0.0;
                for (int l = 0; l < k; ++l) {
                    sum += localH(j, l) * pivot_row[l];
                }
                // std::cout<<"sum" << sum << std::endl;
                // std::cout<<"localQ" << localQ(j, k) << std::endl;
                // std::cout<<"pivot val"<<pivot_row[k]<<std::endl;
                double result = (localQ(j, k) - sum) / pivot_row[k];
                // double abs_result = std::abs(result);
                // if (abs_result > DBL_MAX) {
                //     result = DBL_MAX * (result < 0 ? -1 : 1);  // Preserve the sign
                // } else if (abs_result < 0.0001) {
                //     result = 0.0001 * (result < 0 ? -1 : 1);  // Preserve the sign
                // }
                localH(j, k) = result;
            } else {
                localH(j, k) = pivot_row[k];
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < rows_for_this_process; ++j) {
            // std::cout<<"local v"<<local_v[j]<<std::endl;
            // std::cout<<j << k << std::endl;
            local_v[j] -= localH(j, k) * localH(j, k);
        }
    }
    return p;
}



MaxLoc find_global_max_index(const std::vector<double>& local_v, int local_size, int start_row,  int rank, int size) {
    double local_max = local_v[0];
    int local_index = 0;
    for (int i = 1; i < local_size; ++i) {
        if (local_v[i] > local_max) {
            local_max = local_v[i];
            local_index = i;
        }
    }
    MaxLoc local_max_loc = {local_max, local_index + start_row};
    MaxLoc global_max_loc;

    // Use MPI_Allreduce to find the global maximum value and index
    MPI_Allreduce(&local_max_loc, &global_max_loc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    // std::cout<<"global_max_loc"<<global_max_loc.value<<std::endl;
    return global_max_loc;
}
