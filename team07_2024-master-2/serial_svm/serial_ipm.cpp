#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <cfloat>

#include "serial_ipm.h"

Eigen::VectorXd PrimalDualIPM::Solve(const Eigen::VectorXd& labels, const Eigen::MatrixXd& Q, const Parameters& params) {
    // std::cout << "Rows in matrix m: " << Q.rows() << std::endl;
    // std::cout << "Columns in matrix m: " << Q.cols() << std::endl;
    // std::cout<< "label size"<< labels.size()<<std::endl;
    
    int local_num_rows = labels.size();
    double c_pos = params.weight_positive * params.hyper_parm;
    double c_neg = params.weight_negative * params.hyper_parm;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd la = Eigen::VectorXd::Constant(local_num_rows, c_pos / 10.0);
    Eigen::VectorXd xi = Eigen::VectorXd::Constant(local_num_rows, c_pos / 10.0);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd y = labels;
    Eigen::VectorXd C = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::MatrixXd sigma;
    Eigen::MatrixXd Sigma_inv;
    Eigen::MatrixXd D;
    Eigen::VectorXd Sigma_inv_y;
    Eigen::VectorXd Sigma_inv_z;
    Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd delta_la = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd delta_xi = Eigen::VectorXd::Zero(local_num_rows);
    double delta_nu = 0.0;
    double nu = 0.0;
    double eta = 0.0;
    double resp = 10000000.0;
    double resd = 10000000.0;
    double t = 0.0;
    double ap = DBL_MAX;
    double ad = DBL_MAX;
    for (int i = 0; i < local_num_rows; ++i) {
        double c = (labels(i) > 0) ? c_pos : c_neg;
        C(i) = c;
        la(i) = c / 10.0;
        xi(i) = c / 10.0;
    }
    for (int step = 0; step < params.max_iter; ++step) {
        eta = ComputeSurrogateGap(c_pos, c_neg, labels, x, la, xi);
        t = (params.mu_factor) * 2 * local_num_rows / eta;
        D = ComputeD(xi, la, x, C, params.epsilon);
        z = ComputeZ(Q,x,nu,y,t,C, params.epsilon);
        sigma = ComputeSigma(D,Q);
        Sigma_inv = (sigma+params.epsilon * Eigen::MatrixXd::Identity(local_num_rows, local_num_rows)).inverse();
        // Update x, la, xi, and nu based on your provided formulas
        delta_x = ComputeDeltaX(z,y,delta_nu,Sigma_inv);
        delta_la = ComputeDeltaLambda(la, x, delta_x,t,C, params.epsilon);
        delta_xi = ComputeDeltaXi(xi, x,delta_x,t, params.epsilon);
        Sigma_inv_y = Sigma_inv*y;
        Sigma_inv_z = Sigma_inv*z;
        delta_nu = ComputeDeltaNu(y,z,x,Sigma_inv_y,Sigma_inv_z);
        resp = y.dot(x);
        resd = (la - xi + z).norm();
        // Check convergence
        // check later; current overflow problems not from here
        if (resp <= params.r_pri && resd <= params.r_dual && eta <= params.sgap) {
            break; // Convergence achieved
        }
        // Perform the updates
        ap = DBL_MAX;
        ad = DBL_MAX;
        for (int i = 0; i < local_num_rows; ++i) {
            double c = (labels(i) > 0.0) ? c_pos : c_neg;
            if (delta_x(i) > 0.0) {
                ap = std::min(ap, (c - x(i)) / delta_x(i));
            }
            if (delta_x(i) < 0.0) {
                ap = std::min(ap, -x(i) / delta_x(i));
            }
            if (delta_xi(i) < 0.0) {
                ad = std::min(ad, -xi(i) / delta_xi(i));
            }
            if (delta_la(i) < 0.0) {
                ad = std::min(ad, -la(i) / delta_la(i));
            }
        }
        ap = std::min(ap, 1.0)*0.99; // Safeguard to maintain strict feasibility
        ad = std::min(ad, 1.0)*0.99;
        x += ap * delta_x;
        xi += ad * delta_xi;
        la += ad * delta_la;
        nu += ad * delta_nu;
        
    }
    return x; // Return the solution vector
}
double PrimalDualIPM::calculate_bias(const Eigen::VectorXd& alphas, const Eigen::VectorXd& labels, const Eigen::MatrixXd& Q) {
    double bias = 0.0;
    int num_support_vectors = alphas.size();

    for (int i = 0; i < num_support_vectors; ++i) {
        if (alphas(i) > 0.0) {  // Check if alpha is nonzero
            double inner_product = 0.0;
            for (int j = 0; j < num_support_vectors; ++j) {
                inner_product += alphas(j) * labels(j) * Q(i, j);  // Compute inner product
            }
            bias += labels(i) - inner_product;  // Update bias
        }
    }
    bias /= num_support_vectors;  // Compute average bias
    return bias;
}

double PrimalDualIPM::predict(const Eigen::VectorXd& alphas, const Eigen::VectorXd& labels, const Eigen::MatrixXd& support_vectors, const Eigen::VectorXd& new_point, double bias, const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>& kernel_function) {
    double decision_value = 0.0;
    int num_support_vectors = alphas.size();

    for (int i = 0; i < num_support_vectors; ++i) {
        decision_value += alphas(i) * labels(i) * kernel_function(support_vectors.row(i), new_point);  // Compute the kernel function value
    }
    decision_value += bias; 
    std::cout<<decision_value<<std::endl;
    return (decision_value >= 0.0) ? 1.0 : -1.0;
}

Eigen::VectorXd PrimalDualIPM::ComputeDeltaLambda(const Eigen::VectorXd& lambda, const Eigen::VectorXd& x, 
                                                  const Eigen::VectorXd& delta_x, double t, const Eigen::VectorXd& C, double epsilon) {
    // Ensure no values in the denominator are less than epsilon for stability
    Eigen::ArrayXd safe_denom = (C.array() - x.array()).max(epsilon);
    Eigen::ArrayXd vec_part = (1.0 / t) * safe_denom.inverse();
    Eigen::VectorXd adjusted_lambda = (lambda.array() / safe_denom).max(epsilon);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> lambda_diag = adjusted_lambda.asDiagonal();
    Eigen::VectorXd diag_part = lambda_diag * delta_x;
    Eigen::VectorXd delta_lambda = (-lambda) + vec_part.matrix() + diag_part;
    return delta_lambda;
}



Eigen::VectorXd PrimalDualIPM::ComputeDeltaXi(const Eigen::VectorXd& xi, const Eigen::VectorXd& x, 
                                              const Eigen::VectorXd& delta_x, double t, double epsilon) {
    // Prevent division by very small values
    Eigen::ArrayXd safe_x = x.array().max(epsilon);
    Eigen::ArrayXd inv_t_x = (1.0 / (t * safe_x));
    Eigen::VectorXd adjusted_xi = (xi.array() / safe_x).max(epsilon);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> xi_diag = adjusted_xi.asDiagonal();
    Eigen::VectorXd term_xi_delta_x = xi_diag * delta_x;
    Eigen::VectorXd result = -xi + inv_t_x.matrix() - term_xi_delta_x;

    return result;
}




double PrimalDualIPM::ComputeDeltaNu(const Eigen::VectorXd& y, const Eigen::VectorXd& z, const Eigen::VectorXd& x,
                                     const Eigen::VectorXd& Sigma_inv_y, const Eigen::VectorXd& Sigma_inv_z) {
    double numerator = y.dot(Sigma_inv_z) + y.dot(x);
    double denominator = y.dot(Sigma_inv_y);
    return numerator / denominator;
}

Eigen::VectorXd PrimalDualIPM::ComputeDeltaX(const Eigen::VectorXd& z, const Eigen::VectorXd& y, 
                                             double delta_nu, const Eigen::MatrixXd& Sigma_inv) {
    return Sigma_inv * (z - y * delta_nu);
}

Eigen::MatrixXd PrimalDualIPM::ComputeD(const Eigen::VectorXd& xi, const Eigen::VectorXd& lambda, 
                                        const Eigen::VectorXd& x, const Eigen::VectorXd& C, double epsilon) {
    Eigen::VectorXd D_vec = (xi.array() / x.array().max(epsilon)).max(epsilon) + (lambda.array() / (C.array() - x.array()).max(epsilon)).max(epsilon);

    // Return as a diagonal matrix
    return D_vec.asDiagonal();
}

Eigen::MatrixXd PrimalDualIPM::ComputeSigma(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& D) {
    return D + Q;
}

Eigen::VectorXd PrimalDualIPM::ComputeZ(const Eigen::MatrixXd& Q, const Eigen::VectorXd& x, const double nu, 
                                        const Eigen::VectorXd& y, double t, const Eigen::VectorXd& C,  double epsilon) {
    Eigen::VectorXd vec_one = Eigen::VectorXd::Ones(x.size());
    // Note: The subtraction of nu*y was corrected to addition, as per the formula provided

    return -Q * x + vec_one - nu * y + (1 / t) * ((1 / x.array().max(epsilon)) - (1 / (C - x).array().max(epsilon))).matrix();
}

double PrimalDualIPM::ComputeSurrogateGap(double c_pos, double c_neg, const Eigen::VectorXd& labels, 
                                          const Eigen::VectorXd& x, const Eigen::VectorXd& la, const Eigen::VectorXd& xi) {
    double sum = 0.0;

    for (int i = 0; i < labels.size(); ++i) {
        double c = (labels(i) > 0.0) ? c_pos : c_neg;
        sum += la(i) * c;
    }
    for (int i = 0; i < x.size(); ++i) {
        sum += x(i) * (xi(i) - la(i));
    }
    return sum;
}