////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#include "kernel.h"

// The Kernel class is abstract and does not have its own implementation.
// Specific kernel functions will be implemented in derived classes.

/**
 * @brief Finds inner product of two vectors
 * 
 * @param x1 
 * @param x2 
 * @return ** double 
 */
double Kernel::inner_product(const std::vector<double>& x1, const std::vector<double>& x2) {

    // check sizes
    if (x1.size() !=x2.size()) {

        throw std::invalid_argument("The two vectors must have the same size");
    }
    
    double inner_product = 0.0;

    for(int i = 0; i < x1.size(); ++i) {

        inner_product += (x1[i] * x2[i]);
    }

    return inner_product;
}


/**
 * @brief find norm 2 squared of vector
 *  ||x||^2 = x[0]^2 + x[1]^2 + ...+x[n-1]^2
 * @param vec 
 * @return ** double 
 */
double Kernel::norm_square(const std::vector<double>& vec) {

    // Initialize the sum of squares
    double sum_of_squares = 0.0;

    // Iterate over the vector and add the square of each element to the sum
    for (double val : vec) {
        sum_of_squares += val * val;
    }

    return sum_of_squares;

}