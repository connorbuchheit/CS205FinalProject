// icf_test.cpp
#include <vector>
#include "icf.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <mpi.h>
#include <omp.h>

// can make with g++ -std=c++11 -Wall -I /shared/software/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-7.3.1/eigen-3.3.7-pwe4bmtruhdeqccai3qasfyctdmb3qlj/include test_icf.cpp serialicf.cpp -o serialICFApp
int main() {
    Eigen::MatrixXd matrix(6, 6);
    int n = 6;
    int p = 3;
    // Initialize matrix elements
    matrix << 1, 2, 3, 4, 5, 6,
              7, 8, 9, 10, 11, 12,
              13, 14, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 24,
              25, 26, 27, 28, 29, 30,
              31, 32, 33, 34, 35, 36;
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = 6 / size;
    int extra_rows = 6 % size; // Calculate the number of extra rows

    int start_row, end_row;

    if (rank < extra_rows) {
        // Processes with rank less than extra_rows get one extra row
        rows_per_process++;
        start_row = rank * rows_per_process;
    } else {
        start_row = rank * rows_per_process + extra_rows;
    }
    end_row = start_row + rows_per_process;
    if (rank == size - 1) {
        end_row = 6;
    }
    Eigen::MatrixXd submatrix = matrix.block(start_row, 0, end_row - start_row, n);
    Eigen::MatrixXd output_submatrix(end_row - start_row, p);
    parallelICF(submatrix,n,p,output_submatrix);
    MPI_Finalize();
    return 0;
}