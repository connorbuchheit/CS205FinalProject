////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#include "linear_kernel.h"
#include "kernel.h"
/**
 * @brief Linear: K(x1,x2)=x1^⊤.x2
 * Linear kernel does not have any hyperparameters
 * @param x1 
 * @param x2 
 * @param params 
 * @return ** double 
 */
double LinearKernel::operator()(const std::vector<double>& x1, 
                                    const std::vector<double>& x2, 
                                    const svm_param& params) {

    // Linear: K(x1,x2)=x1^⊤.x2 (https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote13.html)
    // Linear kernel does not have any hyperparameters
    
    // check sizes (we expect all vectors to have same dimension)
    if (x1.size() != x2.size()) {
        throw std::invalid_argument("Data points passed to linear kernel do not have same dimensions");
    }

    // get inner product of the two data points
    return Kernel::inner_product(x1, x2);

}