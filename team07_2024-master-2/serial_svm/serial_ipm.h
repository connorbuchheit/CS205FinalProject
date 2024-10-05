#ifndef SERIAL_IPM_H
#define SERIAL_IPM_H

#include <Eigen/Dense>

class PrimalDualIPM {
public:
    struct Parameters {
        double weight_positive;
        double weight_negative;
        double hyper_parm;
        int max_iter;
        double mu_factor;
        double r_pri;
        double r_dual;
        double sgap;
        // for numerical reasons?
        double epsilon;
    };

    Eigen::VectorXd Solve(const Eigen::VectorXd& labels, const Eigen::MatrixXd& Q, const Parameters& params);
    Eigen::VectorXd ComputeDeltaLambda(const Eigen::VectorXd& lambda, const Eigen::VectorXd& x, 
                                                  const Eigen::VectorXd& delta_x, double t, const Eigen::VectorXd& C, double epsilon);
    Eigen::VectorXd ComputeDeltaXi(const Eigen::VectorXd& xi, const Eigen::VectorXd& x, 
                                              const Eigen::VectorXd& delta_x, double t, double epsilon);
    double ComputeDeltaNu(const Eigen::VectorXd& y, const Eigen::VectorXd& z, const Eigen::VectorXd& x,
                                     const Eigen::VectorXd& Sigma_inv_y,const Eigen::VectorXd& Sigma_inv_z);

    Eigen::VectorXd ComputeDeltaX(const Eigen::VectorXd& z, const Eigen::VectorXd& y, 
                                             double delta_nu, const Eigen::MatrixXd& Sigma_inv);

    Eigen::MatrixXd ComputeD(const Eigen::VectorXd& xi, const Eigen::VectorXd& lambda, 
                                        const Eigen::VectorXd& x, const Eigen::VectorXd& C,  double epsilon);

    Eigen::MatrixXd ComputeSigma(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& D);

    Eigen::VectorXd ComputeZ(const Eigen::MatrixXd& Q, const Eigen::VectorXd& x, const double nu,
                                        const Eigen::VectorXd& y, double t, const Eigen::VectorXd& C,double epsilon);
    double ComputeSurrogateGap(double c_pos, double c_neg, const Eigen::VectorXd& la, const Eigen::VectorXd& xi,
                               const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    double predict(const Eigen::VectorXd& alphas, const Eigen::VectorXd& labels, const Eigen::MatrixXd& support_vectors, const Eigen::VectorXd& new_point, double bias, const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>& kernel_function);

    double calculate_bias(const Eigen::VectorXd& alphas, const Eigen::VectorXd& labels, const Eigen::MatrixXd& Q);
};


#endif
