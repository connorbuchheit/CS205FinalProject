#include "serial_ipm.h"
#include <iostream>
#include <Eigen/Dense>
// make with 
// g++ -I /shared/software/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-7.3.1/eigen-3.3.7-pwe4bmtruhdeqccai3qasfyctdmb3qlj/include -o PrimalDualIPM 
// test_ipm.cpp serial_ipm.cpp 

int main() {
    // Set up parameters for the test
    PrimalDualIPM::Parameters params;
    params.weight_positive = 1.0; // Assume equal weights for simplicity
    params.weight_negative = 1.0;
    params.hyper_parm = 1.0;
    params.max_iter = 100;
    params.mu_factor = 0.9; // Adjust if needed
    params.epsilon = 0.0001;
    params.r_pri = 1e-4;   // Primal residual threshold
    params.r_dual = 1e-4;  // Dual residual threshold
    params.sgap = 1e-6; 
    // Dimension of the problem
    int n = 2;

    // Labels for a binary classification with a known solution
    Eigen::VectorXd labels(n);
    labels << 1, -1;
    Eigen::MatrixXd Q(n, n);
    Q << 2, 0.1,
         0.1, 2;

    // Initialize the PrimalDualIPM solver
    PrimalDualIPM solver;
    Eigen::VectorXd solution = solver.Solve(labels, Q, params);
    std::cout << "Optimized x: " << solution.transpose() << std::endl;

    return 0;
}
