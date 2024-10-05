#include <eigen3/Eigen/Dense>
#include <functional>
#include <cstring>
#include <stdio.h>
#include <fstream>
#include <numeric> 
#include <omp.h>
#include "iostream"
#include "utils.h"
#include "../serial_svm/kernel.h"
#include "../serial_svm/linear_kernel.h"
#include "../serial_svm/polynomial_kernel.h"
#include "../serial_svm/rbf_kernel.h"
void exit_with_help() {
    printf(
        "* Usage: svm-train [options] training_Set_file\n"
        "* options:\n"
        "*  -k kernel_type: set type of kernel function:\n"
        "*      0 -- linear\n"
        "*      1 -- polynomial (default)\n"
        "*      2 -- rbf\n"
    );
    exit(1);
}
void parse_command_line(svm_param& param, int argc, char **argv, char *features_file_name, char *labels_file_name, char *model_file_name,
                                                                char * test_features_file_name, char* test_labels_file_name,
                                                                char* test_predictions_file_name) {
   
    // Set default kernel type
    param.kernel_type = 1;  // default to polynomial kernel

    // Check for correct number of arguments
    std::cout << "argument count is: " << argc << std::endl;
    if (argc != 8) {
        exit_with_help();
    }

    int file_count = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {  // it's a filename
            if (file_count == 0)
                strcpy(features_file_name, argv[i]);
            else if (file_count == 1)
                strcpy(labels_file_name, argv[i]);
            else if (file_count == 2)
                strcpy(test_features_file_name, argv[i]);
            else if (file_count == 3)
                strcpy(test_labels_file_name, argv[i]);
            else if (file_count == 4)
                strcpy(test_predictions_file_name, argv[i]);
            file_count++;
        } else {  // it's an option
            switch (argv[i][1]) {
                case 'k':  // kernel type
                    if (i < argc - 1) param.kernel_type = atoi(argv[++i]);
                    break;
                default:
                    exit_with_help();
            }
        }
    }

    if (file_count != 5) {  // Ensure exactly five filenames are provided
        exit_with_help();
    }

    // Set model file name based on features file name with _model suffix
    sprintf(model_file_name, "%s_model", features_file_name);

    printf("Features file: %s\n", features_file_name);
    printf("Labels file: %s\n", labels_file_name);
    printf("Model file: %s\n", model_file_name);
    printf("Test Features file: %s\n", test_features_file_name);
    printf("Test Labels file: %s\n", test_labels_file_name);
    printf("Predictions file: %s\n", test_predictions_file_name);
    printf("Kernel type: %d\n", param.kernel_type);
}


/**
 * @brief read input data and form q matrix
 * 
 * @param file_name 
 * @param Q 
 * @return ** void 
 */
void read_q_matrix(svm_param& param, const char* file_name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* Q) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open train data file");
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> values;
        double value;
        char comma;

        // Efficiently read all values separated by commas
        while (iss >> value) {
            values.push_back(value);
            if (!(iss >> comma)) {  // If no comma is found, break the loop
                break;
            }
        }

        data.push_back(values);
    }

    // Determine the number of samples, assumed to be the number of vectors
    int n_samples = data.size();
    if (n_samples == 0) {
        throw std::runtime_error("No data found in file");
    }

    // Resize the Eigen matrix to be a square matrix of the number of samples
    Q->resize(n_samples, n_samples);
    // LinearKernel linear_kernel;
    // PolynomialKernel poly_kernel;
    // RBFKernel rbf_kernel;
    // Compute the Q matrix using the specified kernel function
    KernelFunc kernelFunc = selectKernelFunction(param.kernel_type);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            // switch(param.kernel_type){
            //     // change later when solve the include kernel error
            //     // for now I defined a temporary linearkernel function to use.
            //     case 0:  // Linear kernel
            //         (*Q)(i, j) = linearKernel(data[i], data[j], param);
            //         break;
            //     case 1:  // Polynomial kernel
            //         (*Q)(i, j) = linearKernel(data[i], data[j], param);
            //         break;
            //     case 2:  // RBF kernel
            //         (*Q)(i, j) = linearKernel(data[i], data[j], param);
            //         break;
            //     default:  // Default case should also handle errors or use a default kernel
            //         (*Q)(i, j) = linearKernel(data[i], data[j], param);
            //         break;
            // }
            (*Q)(i, j) = kernelFunc(data[i], data[j], param);
        }
    }
}


// function to directly read features
void read_features_matrix(const char* file_name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* Q) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open train data file");
    }

    std::vector<std::vector<double>> data;
    std::string line;
    int n_features = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> values;
        double value;
        char comma;

        // Read values separated by commas
        while (iss >> value) {
            values.push_back(value);
            iss >> comma;  // Consume the comma if present, ignore if not
        }

        if (!data.empty() && values.size() != data.front().size()) {
            throw std::runtime_error("Inconsistent number of features per row");
        }

        data.push_back(values);
        n_features = values.size();  // Update the number of features based on the first successful read
    }

    if (data.empty()) {
        throw std::runtime_error("No data read from file");
    }

    // std::cout << "Number of features: " << n_features << std::endl;

    // Compute the size of the Q matrix
    int n_samples = data.size();
    (*Q).resize(n_samples, n_features);

    // Copy data to the Eigen matrix
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            (*Q)(i, j) = data[i][j];
            // std::cout<<"Qij:"<<i<<" "<<j<<" "<<(*Q)(i, j)<<std::endl;
        }
    }
}
// Function to read labels from a file
void read_labels(const char* file_name, Eigen::VectorXd& labels) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    std::vector<double> temp_labels;  // Temporary container for the labels

    // Read labels from file
    while (std::getline(file, line)) {
        if (!line.empty()) {
            temp_labels.push_back(std::stod(line));  // Convert string to double and store it
        }
    }

    // Now we know the number of labels, resize the Eigen::VectorXd
    labels.resize(temp_labels.size());

    // Copy values from the vector to the Eigen::VectorXd
    for (size_t i = 0; i < temp_labels.size(); ++i) {
        labels(i) = temp_labels[i];
    }
}


void save_model(const char* model_file_name, const Eigen::VectorXd& x, double b) {
    std::ofstream file(model_file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    // Write the size of the vector
    file << x.size() + sizeof(b) << "\n";

    // Write the elements of the vector
    for (int i = 0; i < x.size(); ++i) {
        file << x(i) << "\n";
    }
    file << b << "\n";
}



// just for temporary usage
double linearKernel(const std::vector<double>& x1, const std::vector<double>& x2, const svm_param& params) {
    if (x1.size() != x2.size()) {
        throw std::invalid_argument("Vectors must be of the same length for dot product.");
    }

    // Calculate the inner product of x1 and x2
    return std::inner_product(x1.begin(), x1.end(), x2.begin(), 0.0);
}



KernelFunc selectKernelFunction(int kernel_type) {
    switch (kernel_type) {
        case 0: return linearKernel;
        // poly, define later
        case 1: return linearKernel;
        // rbf define later
        case 2: return linearKernel;
        default: return linearKernel;
    }
}
