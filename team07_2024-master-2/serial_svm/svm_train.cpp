////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS205 : High Performance Computing for Science and Engineering
//  
//
////////////////////////////////////////////////////////////////////////

#include "svm.h"
#include "serialicf.h"
#include "serial_ipm.h"
#include <cstring>
#include <stdio.h>
#include <fstream>
#include "kernel.h"
#include "linear_kernel.h"
#include "rbf_kernel.h"
#include "polynomial_kernel.h"
#include "iostream"
/**
 * svm-train solves an SVM-Optimization problem to produce a model
 * Usage: svm-train [options] training_set_filename
 * options:
 *  -k kernel_type: set type of kernel function:
 *      0 -- linear
 *      1 -- polynomial
 *      2 -- rbf 
 */


struct svm_param param;

// Returns usage informations
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
// this is just for test
// Should make the kernel functions defined previously to work as well by changing data structures
double linearKernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) {
    return x1.dot(x2);
}

/**
 * @brief function parses command line arguments and saves input file name and model file name
 * model_file_name will by default be set as input_file_name_model
 * 
 * svm-train solves an SVM-Optimization problem to produce a model
 * Usage: svm-train [options] training_set_filename
 * options:
 *  -k kernel_type: set type of kernel function:
 *      0 -- linear
 *      1 -- polynomial(default)
 *      2 -- rbf 
 *
 * @param argc 
 * @param argv 
 * @param input_file_name train data set file name  
 * @param model_file_name file where model will be saved
 * @return ** void 
 */
void parse_command_line(int argc, char **argv, char *features_file_name, char *labels_file_name, char *model_file_name) {
    // Set default kernel type
    param.kernel_type = 1;  // default to polynomial kernel

    // Check for correct number of arguments
    if (argc != 5) {
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

    if (file_count != 2) {  // Ensure exactly two filenames are provided
        exit_with_help();
    }

    // Set model file name based on features file name with _model suffix
    sprintf(model_file_name, "%s_model", features_file_name);

    printf("Features file: %s\n", features_file_name);
    printf("Labels file: %s\n", labels_file_name);
    printf("Model file: %s\n", model_file_name);
    printf("Kernel type: %d\n", param.kernel_type);
}


/**
 * @brief read input data and form q matrix
 * 
 * @param file_name 
 * @param Q 
 * @return ** void 
 */
void read_q_matrix(const char* file_name, Eigen::MatrixXd* Q) {
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
    LinearKernel linear_kernel;
    PolynomialKernel poly_kernel;
    RBFKernel rbf_kernel;
    // Compute the Q matrix using the specified kernel function
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            switch(param.kernel_type){
                case 0:  // Linear kernel
                    (*Q)(i, j) = linear_kernel(data[i], data[j], param);
                    break;
                case 1:  // Polynomial kernel
                    (*Q)(i, j) = poly_kernel(data[i], data[j], param);
                    break;
                case 2:  // RBF kernel
                    (*Q)(i, j) = rbf_kernel(data[i], data[j], param);
                    break;
                default:  // Default case should also handle errors or use a default kernel
                    (*Q)(i, j) = poly_kernel(data[i], data[j], param);
                    break;
            }
        }
    }
}


// function to directly read features
void read_features_matrix(const char* file_name, Eigen::MatrixXd* Q) {
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

    std::cout << "Number of features: " << n_features << std::endl;

    // Compute the size of the Q matrix
    int n_samples = data.size();
    (*Q).resize(n_samples, n_features);

    // Copy data to the Eigen matrix
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            (*Q)(i, j) = data[i][j];
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



int main(int argc, char **argv) {

    char features_file_name[MAX_FILE_NAME_SIZE];
    char labels_file_name[MAX_FILE_NAME_SIZE];
    char model_file_name[2 * MAX_FILE_NAME_SIZE];

    // parse command line
    parse_command_line(argc, argv, features_file_name, labels_file_name, model_file_name);
    // we also need original features for final computation
    Eigen::MatrixXd features;
    read_features_matrix(features_file_name, &features);
    // Read from input file to get matrix Q
    Eigen::MatrixXd Q;
    read_q_matrix(features_file_name, &Q);  // Use the function to read Q from the input file

    int n = Q.rows();  // Assuming Matrix is a type that has a rows() method
    int p = Q.cols();  // Assuming Matrix is a type that has a cols() method

    // Call ICF for factorization
    // Matrix H;
    // serialICF(Q, n, 0.5 * p, H); // TODO: Determine p correctly

    // Perform Primal-Dual Interior Point Method
    PrimalDualIPM::Parameters ipm_params;  // Create an instance of Parameters
    // TODO: Set the fields of params as needed
    ipm_params.weight_positive = 1.0; // Assume equal weights for simplicity
    ipm_params.weight_negative = 1.0;
    ipm_params.hyper_parm = 1.0;
    ipm_params.max_iter = 100;
    ipm_params.mu_factor = 0.9; // Adjust if needed
    ipm_params.epsilon = 0.0001;
    ipm_params.r_pri = 1e-4;   // Primal residual threshold
    ipm_params.r_dual = 1e-4;  // Dual residual threshold
    ipm_params.sgap = 1e-6; 
    PrimalDualIPM ipmSolver;  // Create an instance of PrimalDualIPM
    

    // Initialize labels vector
    Eigen::VectorXd labels = Eigen::VectorXd::Zero(n);
    // Read labels from the file
    read_labels(labels_file_name, labels);
    Eigen::VectorXd x = ipmSolver.Solve(labels, Q, ipm_params);  // Call the Solve method of PrimalDualIPM
    double b = ipmSolver.calculate_bias(x, labels, Q);
    std::cout<<b<<std::endl;
    // Save the solution to the model file
    // save_model(model_file_name, x,b);  // Assuming you have a function to save the solution
    // one can add additional separate test functions, but for this milestone I just test here
    Eigen::MatrixXd test_data(50, 3);
     test_data << 1.58324215, 1.94373317, 1.00000000, 
-0.13619610, 3.64027081, 1.00000000, 
0.20656441, 1.15825263, 1.00000000, 
2.50288142, 0.75471191, 1.00000000, 
0.94204778, 1.09099239, 1.00000000, 
2.55145404, 4.29220801, 1.00000000, 
2.04153939, 0.88207455, 1.00000000, 
2.53905832, 1.40384030, 1.00000000, 
1.98086950, 3.17500122, 1.00000000, 
1.25212905, 2.00902525, 1.00000000, 
1.12189211, 1.84356583, 1.00000000, 
2.25657045, 1.01122095, 1.00000000, 
1.66117803, 1.76381597, 1.00000000, 
1.36234499, 0.81238771, 1.00000000, 
0.57878277, 1.84650480, 1.00000000, 
1.73094304, 4.23136679, 1.00000000, 
-0.43476758, 2.11272650, 1.00000000, 
2.37044454, 3.35963386, 1.00000000, 
2.50185721, 1.15578630, 1.00000000, 
2.00000976, 2.54235257, 1.00000000, 
1.68649180, 2.77101174, 1.00000000, 
0.13190935, 3.73118467, 1.00000000, 
3.46767801, 1.66432266, 1.00000000, 
2.61134078, 2.04797059, 1.00000000, 
1.17086471, 2.08771022, 1.00000000, 
-0.99963411, -2.38109252, -1.00000000, 
-2.37566942, -2.07447076, -1.00000000, 
-1.56650367, -0.72162077, -1.00000000, 
-2.63467931, -1.49160376, -1.00000000, 
-1.78388399, -3.85861239, -1.00000000, 
-2.41931648, -2.13232890, -1.00000000, 
-2.03957024, -1.67399657, -1.00000000, 
-4.04032305, -1.95374448, -1.00000000, 
-2.67767558, -3.43943903, -1.00000000, 
-1.47570357, -1.26472042, -1.00000000, 
-2.65325027, -1.15754372, -1.00000000, 
-2.38151648, -1.93351099, -1.00000000, 
-3.09873895, -0.41551294, -1.00000000, 
-4.65944946, -2.09145262, -1.00000000, 
-1.30488039, -4.03346655, -1.00000000, 
-2.18946926, -2.07721867, -1.00000000, 
-1.17529699, -0.75178708, -1.00000000, 
-2.40389227, -3.38451867, -1.00000000, 
-0.63276458, -0.78211437, -1.00000000, 
-2.46200535, -1.64911151, -1.00000000, 
-1.61813377, -1.43372456, -1.00000000, 
-1.79579202, -0.59330376, -1.00000000, 
-3.73795950, -0.95917605, -1.00000000, 
-1.61952803, -2.21713527, -1.00000000, 
-0.82646850, -4.34360319, -1.00000000;

    Eigen::VectorXd test_labels = test_data.col(2); // Extract labels
    Eigen::MatrixXd test_features = test_data.block(0, 0, 50, 2); // Extract features
    int correct_predictions = 0;
    for (int i = 0; i < test_data.rows(); ++i) {
        double predicted_label = ipmSolver.predict(x, labels,features, test_features.row(i), b, linearKernel);
        if (predicted_label == test_labels(i)) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / test_data.rows();
    std::cout << "Accuracy of predictions: " << accuracy * 100 << "%" << std::endl;
}
