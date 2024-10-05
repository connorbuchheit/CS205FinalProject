
////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#ifndef KERNEL_H
#define KERNEL_H

#include "svm.h"

#include<unordered_map>
#include<string>
#include <vector>
#include <stdexcept> // using this for not implemented errors. TODO remove after implementation

/**
 * @brief The kernel class is abstract and does not have its own implementation
 * Specific kernel functions will be implemented in derived classes
 */


class Kernel {

public:
    virtual ~Kernel() {}

    // This virtual function must be overridden in the derived classes of kernel functions implementation
    // It takes in two vectors representing two data points(of length dimension of the data) and returns a double
    // The third parameter is a map which holds the parameters for each kernel function
    // For example, for an RBF kernel, you might call the function like this:
        // std::unordered_map<std::string, double> params = {{"gamma", 0.1}};
        // double result = rbfKernel(vec1, vec2, params);
        // And for a polynomial kernel:
        // std::unordered_map<std::string, double> params = {{"degree", 3}};
        // double result = polyKernel(vec1, vec2, params);
    // For instance, the RBF kernel would use params["gamma"], and the polynomial kernel would use params["degree"]
    virtual double operator() (const std::vector<double>& x1, 
                                const std::vector<double>& x2,
                               const svm_param& params) = 0;

    // This implements inner product of two vectors. This is used in linear and polynomial kernels
    double inner_product(const std::vector<double>& x1, const std::vector<double>& x2);

    // This find the norm 2 squared of a vector
    double norm_square(const std::vector<double>& vec);
};

#endif //KERNEL_H