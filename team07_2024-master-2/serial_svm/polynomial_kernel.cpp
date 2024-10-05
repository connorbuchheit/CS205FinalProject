////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#include "polynomial_kernel.h"
#include "kernel.h"
/**
 * @brief polynomial kernel function
 * K(X, Y) = (gamma <x1, x2> + coef_const) ^ degree
 * gamma = 1/x1.size()
 * coeff_const default is 1.0
 * check https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html for formula
 * @param x1 
 * @param x2 
 * @param params 
 * @return ** double 
 */
double PolynomialKernel::operator()(const std::vector<double>& x1, 
                                        const std::vector<double>& x2,
                                        const svm_param& params) {
   
    // check sizes (we expect all vectors to have same dimension)
    if (x1.size() != x2.size()) {
        throw std::invalid_argument("Data points passed to polynomial kernel do not have same dimensions");
    }

    if (x1.size() == 0) {
        throw std::invalid_argument("data dimension cannot be zero");
    }
    double coeff_const = 1.0; // we can certainly check if this needs to be changed
    double gamma = 1 / x1.size();

    // find degree hyperparameter
    double degree = params.degree;

    double inner_prod_plus_coeff = coeff_const + gamma * Kernel::inner_product(x1, x2);

    return std::pow(inner_prod_plus_coeff, degree);    

}