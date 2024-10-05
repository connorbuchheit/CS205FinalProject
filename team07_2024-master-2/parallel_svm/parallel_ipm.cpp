#include <iostream>
#include <limits>
#include <cfloat>
#include <vector> 
#include <chrono> 

#include "parallel_ipm.h"

MetricsCollector metrics_;

void LinearSolveViaICFCol(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& D, const Eigen::VectorXd& b, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lower_ra, int local_num_rows, Eigen::VectorXd& x)
{
    int process_id, size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    auto startComputation = std::chrono::high_resolution_clock::now();

    // Convert D to a diagonal matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D_matrix = D.asDiagonal();

    // Compute VV'
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> VVt = H * H.transpose();

    // Compute D + VV'
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A = D_matrix + VVt;

    // Compute the Cholesky decomposition of A
    Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lltOfA(A);

    // Solve Ax = b for x
    if(lltOfA.info() == Eigen::Success) {
        x = lltOfA.solve(b);
    } else {
        // std::cout << "Cholesky decomposition failed. The matrix might not be positive definite." << std::endl;
    }
    // Stop timing computation
    auto endComputation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);

}

// Parallel Primal Dual Interior Point Method
// Matrix H is distributedly in the m machines at the end of PICF
// We will distribute all the n*1 vectors in a round-robin fashion on the m machines
// These vectors are z, α, ξ, λ, ∆z, ∆α, ∆ξ, and ∆λ.
// Every machine caches a copy of global data including ν, t, n, and ∆ν. 
// Whenever a scalar is changed, a broadcast is required to maintain global consistency.

/**
local_num_rows : num of samples in this processor
global_num_rows: total num of samples in training instance
H : parallel icf result for process
**/
Eigen::VectorXd PrimalDualIPM::Solve(const Eigen::VectorXd& local_labels,
                                     const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H,
                                     const int local_num_rows,
                                     const int global_num_rows,
                                     const Parameters& params) {
 
    int process_id, size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    auto start = std::chrono::high_resolution_clock::now();
    auto startComputation = std::chrono::high_resolution_clock::now();

    // initialize vectors and other consts for training 
    double c_pos = params.weight_positive * params.hyper_parm;
    double c_neg = params.weight_negative * params.hyper_parm;

    Eigen::VectorXd x = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd la = Eigen::VectorXd::Constant(local_num_rows, c_pos / 10.0);
    Eigen::VectorXd xi = Eigen::VectorXd::Constant(local_num_rows, c_pos / 10.0);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd y = local_labels;
    Eigen::VectorXd C = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D;
    Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd delta_la = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd delta_xi = Eigen::VectorXd::Zero(local_num_rows);
    Eigen::VectorXd tlx(local_num_rows), tux(local_num_rows), xilx(local_num_rows), laux(local_num_rows);

    double delta_nu = 0.0;
    double nu = 0.0;

    double eta;
    double resp;
    double resd;
    double t;

    for (int i = 0; i < local_num_rows; ++i) {
        double c = (local_labels(i) > 0) ? c_pos : c_neg;
        C(i) = c;
        la(i) = c / 10.0;
        xi(i) = c / 10.0;
    }


    // get process id running this and log
    int h_cols = H.cols();
    
    if (process_id == 0) {
        std::cout << "IPM solver is running: for H with rank: " << h_cols << std::endl;
    }
 
    // create lower triangular matrix from H
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lower_H = H.triangularView<Eigen::Lower>();

    // Stop timing computation
    auto endComputation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);

    // wait for all
    auto startCommunication = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    auto endCommunication = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> communicationElapsed = endCommunication - startCommunication;

    // Record communication duration
    metrics_.recordCommunicationTime("communication", communicationElapsed/size_);

    for (int step = 0; step < params.max_iter; ++step) {
        if (process_id == 0) {
            // std::cout << "IPM solver iteration: " << step << std::endl;
        }
      
        eta = ComputeSurrogateGap(c_pos, c_neg, local_labels, x, la, xi);
        t = (params.mu_factor) * (2.0 * global_num_rows) / eta;

        //computes z = H H^T \alpha - tradeoff \alpha (we use tradeoff 0)
        ComputePartialZ(H, x, 0,local_num_rows, z);
      
        resp = 0.0;
        resd = 0.0;

        // solve equation 8 and 11 in the paper
        startComputation = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < local_num_rows; ++i) {
            double temp;
            z[i] += nu * local_labels[i] - 1.0;
            temp = la[i] - xi[i] + z[i];
            resd += temp * temp;
            resp += local_labels[i] * x[i];
        }
        double from_sum[2], to_sum[2];
        from_sum[0] = resp;
        from_sum[1] = resd;

        endComputation = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> computationElapsed = endComputation - startComputation;

        // Record computation duration
        metrics_.recordComputationTime("computation", computationElapsed/size_);

        // Use MPI to perform a reduction operation to sum the partial results
        startCommunication = std::chrono::high_resolution_clock::now();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(from_sum, to_sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        resp = fabs(to_sum[0]);
        resd = sqrt(to_sum[1]);
        endCommunication = std::chrono::high_resolution_clock::now();
        communicationElapsed = endCommunication - startCommunication;
        metrics_.recordCommunicationTime("communication", communicationElapsed/size_);

        // Check convergence
        if (resp <= params.r_pri && resd <= params.r_dual && eta <= params.sgap) {
            break; // Convergence achieved
        }

        // updating variables and completing z
        startComputation = std::chrono::high_resolution_clock::now();
        double m_lx, m_ux;
        
        for (int i = 0; i < local_num_rows; ++i) {
            double c = (local_labels[i] > 0) ? c_pos : c_neg;
            m_lx = std::max(x[i], params.epsilon);
            m_ux = std::max(c - x[i], params.epsilon);
            tlx[i] = 1.0 / (t * m_lx);
            tux[i] = 1.0 / (t * m_ux);
            xilx[i] = std::max(xi[i] / m_lx, params.epsilon);
            laux[i] = std::max(la[i] / m_ux, params.epsilon);
            d[i] = 1.0 / (xilx[i] + laux[i]);  
        }

        // complete z wth intermediates above
        for (int i = 0; i < local_num_rows; ++i)
            z[i] = tlx[i] - tux[i] - z[i];

        // Perform the newton step
        // calculate lower_H as E = I+H^T D H
        int dim_ = lower_H.cols();
        lower_H = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(dim_, dim_);
        endComputation = std::chrono::high_resolution_clock::now();
        computationElapsed = endComputation - startComputation;
        // Record computation duration
        metrics_.recordComputationTime("computation", computationElapsed/size_);

        ComputeLowerH(H, d, lower_H);
       
        // Check if the process_id is 0
        startComputation = std::chrono::high_resolution_clock::now();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lower_ra;
        if (process_id == 0) {

            // Perform Cholesky factorization
            Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lltOfLowerH(lower_H);
            if(lltOfLowerH.info() == Eigen::Success) {
                // If the factorization is successful, save it in lower_ra
                lower_ra = lltOfLowerH.matrixL();
            } else {
                // std::cout << "Cholesky factorization failed. The matrix might not be positive definite." << std::endl;
            }
        }

        endComputation = std::chrono::high_resolution_clock::now();
        computationElapsed = endComputation - startComputation;
        // Record computation duration
        metrics_.recordComputationTime("computation", computationElapsed/size_);
        
        ComputeDeltaNu(H, d, y, z, x, lower_ra, local_num_rows, delta_nu);
        ComputeDeltaX(H, d, y, delta_nu, lower_ra, z, local_num_rows, delta_x);

        // Update dxi and dla
        startComputation = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < local_num_rows; ++i) {
            delta_xi[i] = tlx[i] - xilx[i] * delta_x[i] - xi[i];
            delta_la[i] = tux[i] + laux[i] * delta_x[i] - la[i];
        }

        // Line Search
        // line search for primal and dual variable
        double ap = DBL_MAX;
        double ad = DBL_MAX;
        for (int i = 0; i < local_num_rows; ++i) {
            // make sure alpha + delta*alpha is in [epsilon, C - epsilon],
            double c = (y[i] > 0.0) ? c_pos : c_neg;
            if (delta_x[i]  > 0.0) {
                ap = std::min(ap, (c - x[i]) / delta_x[i]);
            }
            if (delta_x[i]  < 0.0) {
                ap = std::min(ap, -x[i]/delta_x[i]);
            }
            // std::cout << "ap in line search loop is: " << ap << " delta x_i is"<< delta_x[i] << " x is: "<< x[i] << " c is" << c <<std::endl;
            
            // make sure xi+ delta*xi is in [epsilon, +inf), also
            // lambda + delta*lambda is in [epsilon, +inf).
            if (delta_xi[i] < 0.0) {
                ad = std::min(ad, -xi[i] / delta_xi[i]);
            }
            if (delta_la[i] < 0.0) {
                ad = std::min(ad, -la[i] / delta_la[i]);
            }
        }
        double from_step[2], to_step[2];
        from_step[0] = ap;
        from_step[1] = ad;
        endComputation = std::chrono::high_resolution_clock::now();
        computationElapsed = endComputation - startComputation;
        // Record computation duration
        metrics_.recordComputationTime("computation", computationElapsed/size_);

        // Use MPI to perform a reduction operation to find the minimum of the partial results
        startCommunication = std::chrono::high_resolution_clock::now();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(from_step, to_step, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        auto endCommunication = std::chrono::high_resolution_clock::now();
        communicationElapsed = endCommunication - startCommunication;

        // Record computation duration
        metrics_.recordCommunicationTime("communication", communicationElapsed/size_);

        // According to Primal-Dual IPM, the solution must be strictly feasible
        // to inequality constraints, here we add some disturbance to avoid
        // equality
        startComputation = std::chrono::high_resolution_clock::now();

        ap = std::min(to_step[0], 1.0) * 0.99;
        ad = std::min(to_step[1], 1.0) * 0.99;

        // Update vectors alpha, xi, lambda, and scalar nu according to Newton
        // step and search direction. This completes one Newton's iteration.
        for (int i = 0; i < local_num_rows; ++i) {
            x[i]  += ad * delta_x[i]; // this should be ap remember to change
            xi[i] += ad * delta_xi[i];
            la[i] += ad * delta_la[i];
        }
      
        nu += ad * delta_nu;  
        endComputation = std::chrono::high_resolution_clock::now();
        computationElapsed = endComputation - startComputation;
        // Record computation duration
        metrics_.recordComputationTime("computation", computationElapsed/size_);      
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed in IPM Solver: " << elapsed.count() << " seconds"  << " process id" << process_id << std::endl;
    if (process_id == 0){
        metrics_.printMetrics();
    }
    return x; // Return the solution vector
}

// Compute Newton direction of primal variable alpha
void PrimalDualIPM::ComputeDeltaX(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& D, const Eigen::VectorXd& y, double dnu, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lower_ra, const Eigen::VectorXd& z, int local_num_rows, Eigen::VectorXd& dx)
{
    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
 
    // Calculate tz = z - y * dnu
    Eigen::VectorXd tz = z - dnu * y;

    // Calculate inv(Q + D) * (z - y * dnu)
    LinearSolveViaICFCol(H, D, tz, lower_ra, local_num_rows, dx);

}


// Compute Newton direction of primal variable nu
void PrimalDualIPM::ComputeDeltaNu(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& D, const Eigen::VectorXd& y, const Eigen::VectorXd& z, const Eigen::VectorXd& x, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lower_ra, int local_num_rows, double& dnu)
{
    
    int process_id, size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
  
    // Calculate inv(Q + D) * z
    Eigen::VectorXd tw(local_num_rows);
    LinearSolveViaICFCol(H, D, z, lower_ra, local_num_rows, tw);

    // Calculate inv(Q + D) * y
    Eigen::VectorXd tl(local_num_rows);
    LinearSolveViaICFCol(H, D, y, lower_ra, local_num_rows, tl);

    auto startComputation = std::chrono::high_resolution_clock::now();
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < local_num_rows; ++i) {
        sum1 += y[i] * (tw[i] + x[i]);
        sum2 += y[i] * tl[i];
    }

    double from_sum[2] = {sum1, sum2};
    double to_sum[2];

    // Stop timing computation
    auto endComputation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);

    // Use MPI to perform a reduction operation to sum the partial results
    auto startCommunication = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(from_sum, to_sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    auto endCommunication = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> communicationElapsed = endCommunication - startCommunication;

    // Record computation duration
    metrics_.recordCommunicationTime("communication", communicationElapsed/size_);

    dnu = to_sum[0] / to_sum[1];

}


void PrimalDualIPM::ComputeLowerH(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& D, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& lower_H)
{
    int process_id, size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    
    auto startComputation = std::chrono::high_resolution_clock::now();
    int rows = H.rows();
    int cols = H.cols();
  
    std::vector<double> buff(std::max(rows, (cols + 1) * cols / 2));
    std::vector<double> result((cols + 1) * cols / 2);
    int offset = 0;
    for (int i = 0; i < cols; ++i) {
        offset += i;
        for (int p = 0; p < rows; ++p) {
            buff[p] = H(p, i) * D[p];
        }
        for (int j = 0; j <= i; ++j) {
            double tmp = 0;
            for (int p = 0; p < rows; ++p) {
                tmp += buff[p] * H(p, j);
            }
            result[offset+j] = tmp;
        }
    }
    auto endComputation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);

    // Use MPI to perform a reduction operation to sum the partial results
    auto startCommunication = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(result.data(), buff.data(), (cols + 1) * cols / 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    auto endCommunication = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> communicationElapsed = endCommunication - startCommunication;

    // Record computation duration
    metrics_.recordCommunicationTime("communication", communicationElapsed/size_);

    startComputation = std::chrono::high_resolution_clock::now();
    if (process_id == 0) {
        int disp = 0;
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j <= i; ++j) {
                // std::cout << "i is: " << i << "and j is: " << j << " lower_ has rows :" << lower_H.rows() << "and cols" << lower_H.cols() << std::endl;
                lower_H(i, j) = buff[disp++] + (i == j ? 1 : 0);
            }
        }
    }

    endComputation = std::chrono::high_resolution_clock::now();
    computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);
}


double PrimalDualIPM::calculate_bias(const Eigen::VectorXd& alphas, const Eigen::VectorXd& labels, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& Q) {
    int process_id, size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    auto start = std::chrono::high_resolution_clock::now();
    
    double bias = 0.0;
    int num_support_vectors = alphas.size();

    auto startComputation = std::chrono::high_resolution_clock::now();
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

    auto endComputation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time elapsed in Calculate Bias: " << elapsed.count() << " seconds"  << " process id" << process_id << std::endl;

    return bias;
}

// computes z = H H^T \alpha - tradeoff \alpha (we use tradeoff 0)
void PrimalDualIPM::ComputePartialZ(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& H, const Eigen::VectorXd& alpha, double tradeoff, int local_num_rows, Eigen::VectorXd& z)
{
    int process_id, size_;
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    auto startComputation = std::chrono::high_resolution_clock::now();

    int p = H.cols();
    // long long flops = 2 * local_num_rows * p + (2 * p + 2) * local_num_rows; // Assuming each operation involves a multiply and add
    // long long bytes = 2 * p * local_num_rows * sizeof(double) + p * sizeof(double) + 2 * (p+1) * local_num_rows * sizeof(double);
    Eigen::VectorXd vz(p);
    Eigen::VectorXd vzpart(p);
    vzpart.setZero();

    // form vz = H^T * alpha
    for (int j = 0; j < p; ++j) {
        double sum = 0.0;
        for (int i = 0; i < local_num_rows; ++i) {
            sum += H(i, j) * alpha[i];
        }
        vzpart[j] = sum;
    }

    auto endComputation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);


    // Use MPI to perform a reduction operation to sum the partial results
    auto startCommunication = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(vzpart.data(), vz.data(), p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    auto endCommunication = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> communicationElapsed = endCommunication - startCommunication;

    // Record communication duration
    metrics_.recordCommunicationTime("communication", communicationElapsed/size_);
    
    startComputation = std::chrono::high_resolution_clock::now();
    // form z = H * vz - tradeoff * alpha
    for (int i = 0; i < local_num_rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < p; ++j) {
            sum += H(i, j) * vz[j];
        }
        z[i] = sum - tradeoff * alpha[i];
    }
    endComputation = std::chrono::high_resolution_clock::now();
    computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);
}



double PrimalDualIPM::ComputeSurrogateGap(double c_pos, double c_neg, const Eigen::VectorXd& labels,
                               const Eigen::VectorXd& x, const Eigen::VectorXd& la, const Eigen::VectorXd& xi)
{
    int process_id, size_;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    int local_num_rows = labels.size();
    double sum = 0.0;

    auto startComputation = std::chrono::high_resolution_clock::now();
   
    // sgap = -<f(x), [la,xi]>
    for (int i = 0; i < local_num_rows; ++i) {
        double c = (labels[i] > 0.0) ? c_pos : c_neg;
        sum += la[i] * c;
    }
    for (int i = 0; i < local_num_rows; ++i) {
        sum += x[i] * (xi[i] - la[i]);
    }

    auto endComputation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> computationElapsed = endComputation - startComputation;

    // Record computation duration
    metrics_.recordComputationTime("computation", computationElapsed/size_);

    double global_sum = 0.0;
    auto startCommunication = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    auto endCommunication = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> communicationElapsed = endCommunication - startCommunication;

    // Record computation duration
    metrics_.recordCommunicationTime("communication", communicationElapsed/size_);

    return global_sum;
}
