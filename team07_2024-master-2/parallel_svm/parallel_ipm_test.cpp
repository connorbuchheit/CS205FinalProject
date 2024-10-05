#include <vector>
#include "icf.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <iomanip>
#include "parallel_ipm.h"
#include "utils.h"
#include "../serial_svm/svm.h"

// this is just for test
// Should make the kernel functions defined previously to work as well by changing data structures
double linearKernels(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) {
    return x1.dot(x2);
}

// can make with g++ -std=c++11 -Wall -I /shared/software/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-7.3.1/eigen-3.3.7-pwe4bmtruhdeqccai3qasfyctdmb3qlj/include test_icf.cpp serialicf.cpp -o serialICFApp
Eigen::VectorXd generateLocalLabels(int start_row, int end_row) {
    int num_rows = end_row - start_row;
    Eigen::VectorXd local_labels(num_rows);

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    // Assign labels randomly as 1 or -1
    for (int i = 0; i < num_rows; ++i) {
        int label = dis(gen) == 0 ? -1 : 1; // Randomly choose between -1 and 1
        local_labels[i] = label;
    }

    return local_labels;
}
void printMatrix(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat, int rank) {
    std::cout << "Process " << rank << " received matrix:" << std::endl;
    std::cout << mat << std::endl;
}
void printMatrixInfo(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat, int rank) {
    std::cout << "Process " << rank << " has a matrix of shape: " 
              << mat.rows() << "x" << mat.cols() << std::endl;
}
void resizeMatrix(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matrix, int new_cols) {
    if (new_cols < matrix.cols()) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> resized = matrix.leftCols(new_cols);
        matrix.swap(resized); 
    }
}

void predict(const char* test_file, 
            const char* test_labels,
            const char* predict_file,
            const Eigen::VectorXd& alphas, 
            Eigen::VectorXd& label,
            int test_size, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& support_vectors,
            const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>& kernel_function,
            double* accuracy, double bias) 
{

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Eigen::VectorXd global_predict;

    if (rank == 0){
        // allocate memory for global prediction
        // this will be stored in master
        global_predict = Eigen::VectorXd::Zero(test_size);
    }

    // every process gets a local prediction
    Eigen::VectorXd local_predict = Eigen::VectorXd::Zero(test_size);
    Eigen::VectorXd test_label = Eigen::VectorXd::Zero(test_size);

    int true_positive_num = 0;
    int false_negative_num = 0;
    int false_positive_num = 0;
    int true_negative_num = 0;

    int processed_samples = 0;

    // open file for reading test data
    std::ifstream file(test_file);
    //std::cout << "test file name is: " << test_file << std::endl;
    if (!file.is_open()) {
        throw std::runtime_error("Could not test data file");
    }

    // Open the label file for reading
    std::ifstream labelFile(test_labels);
    if (!labelFile.is_open()) {
        throw std::runtime_error("Could not open label file");
    }

    std::string labelLine;

    // open predict write file
    std::ofstream predictFile;
    if (rank == 0) {
        predictFile.open(predict_file);
        if (!predictFile.is_open()) {
            throw std::runtime_error("Could not open predict file");
        }
    }

    std::string line;

    while (std::getline(file, line) && std::getline(labelFile, labelLine)) {
        std::istringstream iss(line);
        Eigen::VectorXd values;
        double value;
        char comma;

        // Count the number of values in the line
        int numValues = std::count(line.begin(), line.end(), ',') + 1;

        // Resize the Eigen vector to hold the values
        values.resize(numValues);

        // Read all values separated by commas
        int i = 0;
        while (iss >> value) {
            values(i) = value;
            ++i;
            if (!(iss >> comma)) {  // If no comma is found, break the loop
                break;
            }
        }

        if (processed_samples == (test_size - 1)) {

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Reduce(local_predict.data(), global_predict.data(),processed_samples, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                // Print the sizes of your vectors and the value of processed_samples
                std::cout << "Size of global_predict: " << global_predict.size() << std::endl;
                std::cout << "Size of label: " << test_label.size() << std::endl;
                std::cout << "Value of processed_samples: " << processed_samples << std::endl;
                std::cout << "Bias is : " << bias << std::endl;

                for (int i = 0; i < processed_samples; i++) {
                    double predict_val = global_predict(i) + (bias * 1.0)/size;
                    double predicted_label =  (predict_val > 0.0) ? 1.0 : -1.0;

                    //std::cout<< "i is: " << i << "predicted val is: " << predict_val << " predicted label is: " << predicted_label << " actual label is: " << test_label(i) << "\n" << std::flush;

                    // Write predicted label to predict_file
                    predictFile << predicted_label << "\n" << std::flush;;

                    // Update the true positive, false negative, false positive, and true negative counts
                    if (test_label(i) == 1 && predicted_label == 1) {
                        ++true_positive_num;
                    } else if (test_label(i) == 1 && predicted_label == -1) {
                        ++false_negative_num;
                    } else if (test_label(i) == -1 && predicted_label == 1) {
                        ++false_positive_num;
                    } else if (test_label(i) == -1 && predicted_label == -1) {
                        ++true_negative_num;
                    }
                }

                predictFile.close();
            } 

            //std::cout << "closed predicted file \n" << std::flush;
            break;  
        }

        // read test sample label into test_label(processed_samples)
        test_label(processed_samples) = std::stod(labelLine);

        //std::cout << "predicted samples number is " << processed_samples << std::endl;
        // predict
        double decision_value = 0.0;
        int num_support_vectors = alphas.size();

        for (int i = 0; i < num_support_vectors; ++i) {
            decision_value += alphas(i) * label(i) * kernel_function(support_vectors.row(i), values);  // Compute the kernel function value
            //std::cout << "decision value is : " << decision_value << std::endl;
        }

        local_predict(processed_samples) = decision_value;

        // next sample
        ++processed_samples;
    }
    
    int correct = true_positive_num + true_negative_num;
    int incorrect = false_positive_num + false_negative_num;
    (*accuracy) = static_cast<double>(correct) / (correct + incorrect);
}
            

int main(int argc, char **argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    svm_param param;
    int n = 0;
    int p = 0;
    int m = 0; // number of cols in features
    int rank, size;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> features;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Q;
    Eigen::VectorXd labels;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set up parameters for the test
    // move here so we can pass param into 
    PrimalDualIPM::Parameters params;
    params.weight_positive = 1.0; // Assume equal weights for simplicity
    params.weight_negative = 1.0;
    params.hyper_parm = 1.0;
    params.max_iter = 10;
    params.mu_factor = 0.9; // Adjust if needed
    params.epsilon = 0.0001;
    params.r_pri = 1e-4;   // Primal residual threshold
    params.r_dual = 1e-4;  // Dual residual threshold
    params.sgap = 1e-6;
    char features_file_name[MAX_FILE_NAME_SIZE];
    char labels_file_name[MAX_FILE_NAME_SIZE];
    char test_features_file_name[MAX_FILE_NAME_SIZE];
    char test_labels_file_name[MAX_FILE_NAME_SIZE];
    char test_predictions_file_name[MAX_FILE_NAME_SIZE];
    char model_file_name[2 * MAX_FILE_NAME_SIZE];
    parse_command_line(param, argc, argv, features_file_name, labels_file_name, model_file_name, test_features_file_name,
                    test_labels_file_name, test_predictions_file_name);
    auto start = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        read_features_matrix(features_file_name, &features);
        read_q_matrix(param, features_file_name, &Q);
        // std::cout<<"PSD check"<<isPSD(Q)<<std::endl;
        n = Q.rows();
        // p = 13;
        // p = ceil(sqrt(Q.cols()));
        p = Q.cols();
        double epsilon = 1e-10; 
        Q = Q + epsilon * Eigen::MatrixXd::Identity(n, n);
        // checkPSD(Q);
        m = features.cols();
        read_labels(labels_file_name, labels);

    }
    // Broadcast n and p to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = n / size;
    int extra_rows = n % size;
    int rows_for_this_process = rows_per_process + (rank < extra_rows ? 1 : 0);

    // Allocate space for each process
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> localFeatures(rows_for_this_process, m);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> localQ(rows_for_this_process, n);
    Eigen::VectorXd localLabels(rows_for_this_process);

    // Prepare scatter parameters
    std::vector<int> Qsendcounts(size), Lsendcounts(size), Fsendcounts(size), Qdispls(size),Ldispls(size),Fdispls(size);

    int Qoffset = 0;
    int Loffset = 0;
    int Foffset = 0;
    for (int i = 0; i < size; ++i) {
        Qsendcounts[i] = (rows_per_process + (i < extra_rows ? 1 : 0)) * n;
        Qdispls[i] = Qoffset;
        Qoffset += Qsendcounts[i];
        Lsendcounts[i] = (rows_per_process + (i < extra_rows ? 1 : 0))* 1;
        Ldispls[i] = Loffset;
        Loffset += Lsendcounts[i];
        Fsendcounts[i] = (rows_per_process + (i < extra_rows ? 1 : 0)) * m;
        Fdispls[i] = Foffset;
        Foffset += Fsendcounts[i];
    }

    MPI_Scatterv(features.data(), Fsendcounts.data(), Fdispls.data(), MPI_DOUBLE,
                 localFeatures.data(), Fsendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(Q.data(), Qsendcounts.data(), Qdispls.data(), MPI_DOUBLE,
                 localQ.data(), Qsendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatterv(labels.data(), Lsendcounts.data(), Ldispls.data(), MPI_DOUBLE,
                 localLabels.data(), Lsendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output_submatrix(rows_for_this_process, p);
    double real_p = parallelICF(localQ,n,p,output_submatrix);
    if (real_p < p){
        resizeMatrix(output_submatrix , real_p);
    }
   
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize after output
    PrimalDualIPM solver;

    // Call the Solve method
    Eigen::VectorXd opt_x = solver.Solve(localLabels, output_submatrix, rows_for_this_process, n, params);
   
    double b = solver.calculate_bias(opt_x, localLabels, localQ);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Optimized x: " << opt_x.transpose() << std::endl;

    // predict 
    double accuracy = 0;

    printf("Test Features file: %s\n", test_features_file_name);
    printf("Test Labels file: %s\n", test_labels_file_name);
    printf("Predictions file: %s\n", test_predictions_file_name);

    std::ifstream t_file(test_features_file_name);

    if (!t_file.is_open()) {
        std::cerr << "Could not open the test features file." << std::endl;
        return 1;
    }

    int line_count = 0;
    std::string line;
    while (std::getline(t_file, line)) {
        ++line_count;
    }

    std::cout << "Number of lines: " << line_count << std::endl;

    predict(test_features_file_name, test_labels_file_name , test_predictions_file_name, opt_x, localLabels,line_count,localFeatures, linearKernels, &accuracy, b);

    if (rank == 0) {
        std::cout << "accuracy is: " << accuracy << std::flush;
        std::cout << "Time elapsed in train ipp + icf: " << elapsed.count() << " seconds"  << " process id" << rank << std::endl;
    }

    MPI_Finalize();
    return 0;
}

