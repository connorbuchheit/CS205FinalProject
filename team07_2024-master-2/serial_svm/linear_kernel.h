////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#ifndef LINEAR_KERNEL_H
#define LINEAR_KERNEL_H

#include "kernel.h"

class LinearKernel: public Kernel {

public:
    double operator()(const std::vector<double>& x1, 
                        const std::vector<double>& x2, 
                        const svm_param& params) override;
};

#endif // LINEAR_KERNEL_H