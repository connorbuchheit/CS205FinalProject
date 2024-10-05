#ifndef PARALLEL_IPM_H
#define PARALLEL_IPM_H

#include <eigen3/Eigen/Dense>
#include <mpi.h>
#include "utils.h"

class PrimalDualIPM {
public:
    struct Parameters {
        double weight_positive;
        double weight_negative;
        double hyper_parm;
        int max_iter;
        double mu_factor;
        double epsilon;
        double r_pri;
        double r_dual;
        double sgap;
    };

    Eigen::VectorXd Solve(const Eigen::VectorXd& local_labels,
                          const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H,
                          const int local_num_rows,
                          const int global_num_rows,
                          const Parameters& params);


    double ComputeSurrogateGap(double c_pos, double c_neg, const Eigen::VectorXd& labels,
                               const Eigen::VectorXd& x, const Eigen::VectorXd& la, const Eigen::VectorXd& xi);

    void ComputePartialZ(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& alpha, 
                            double tradeoff, int local_num_rows, Eigen::VectorXd& z);

    void ComputeLowerH(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& d, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lower_H);

    void ComputeDeltaNu(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& d, const Eigen::VectorXd& y,
                        const Eigen::VectorXd& z, const Eigen::VectorXd& x, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lower_ra,
                        const int local_num_rows, double& dnu);

    void ComputeDeltaX(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& d, const Eigen::VectorXd& y,
                       const double dnu, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lower_ra, const Eigen::VectorXd& z,
                       const int local_num_rows, Eigen::VectorXd& dx);
    // double predict(const Eigen::VectorXd& alphas, const Eigen::VectorXd& labels, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& support_vectors,
    //  const Eigen::VectorXd& new_point, double bias, const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>& kernel_function);
    double calculate_bias(const Eigen::VectorXd& alphas, const Eigen::VectorXd& labels, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& Q);



    
};

class MetricsCollector {
private:
    std::map<std::string, double> metrics;

public:
    // Add a metric with a given name and value
    void addMetric(const std::string& name, double value) {
        metrics[name] += value; // If the metric already exists, add the new value to it
    }

    // Record the duration of a computation with a given name
    void recordComputationTime(const std::string& operationName, std::chrono::duration<double> duration) {
        addMetric(operationName + "_time_seconds", duration.count());
    }

    // Record the duration of a communication with a given name
    void recordCommunicationTime(const std::string& operationName, std::chrono::duration<double> duration) {
        addMetric(operationName + "_time_seconds", duration.count());
    }

    // Print all collected metrics
    void printMetrics() const {
        std::cout << "Metrics:" << std::endl;
        for (const auto& metric : metrics) {
            std::cout << metric.first << ": " << metric.second << std::endl;
        }
    }
};



#endif // PARALLEL_IPM_H
