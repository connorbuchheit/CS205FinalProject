////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#ifndef RBF_KERNEL_H
#define RBF_KERNEL_H

#include "kernel.h"

class RBFKernel: public Kernel {

public:
    double operator()(const std::vector<double>& x1, 
                        const std::vector<double>& x2,
                        const svm_param& params) override;
};

#endif // RBF_KERNEL_H