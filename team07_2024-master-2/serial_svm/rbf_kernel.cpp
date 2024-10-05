////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#include "rbf_kernel.h"
#include "kernel.h"
/**
 * @brief K(x,z)=e^((−gamma. ∥x1−x2∥)^2)
 * check https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html for formula
 * @param x1 
 * @param x2 
 * @param params 
 * @return ** double 
 */
double RBFKernel::operator()(const std::vector<double>& x1, 
                                const std::vector<double>& x2,
                                const svm_param& params) {
    // Note: ||a - b||^2 = (||a||^2 - 2 * a.b + ||b||^2)

    // check sizes (we expect all vectors to have same dimension)
    if (x1.size() != x2.size()) {
        throw std::invalid_argument("Data points passed to Gaussian kernel do not have same dimensions");
    }

    // find gamma hyperparameter
    double gamma = params.gamma;
    
    double norm_squared_diff = Kernel::norm_square(x1) - 2* Kernel::inner_product(x1, x2) + Kernel::norm_square(x2);
    double kernel_result = std::exp(-1 * gamma * norm_squared_diff);

    return kernel_result;
}