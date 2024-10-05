////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#ifndef POLYNOMIAL_KERNEL_H
#define POLYNOMIAL_KERNEL_H

#include "kernel.h"

class PolynomialKernel: public Kernel {

public:
    double operator()(const std::vector<double>& x1,
                        const std::vector<double>& x2, 
                        const svm_param& params) override;
};

#endif // POLYNOMIAL_KERNEL_H