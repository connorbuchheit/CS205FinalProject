#ifndef UTILS_H
#define UTILS_H
#include <eigen3/Eigen/Dense>
#include <cstring>
#include <stdio.h>
#include <functional>
#include <fstream>
#include "iostream"
#include "../serial_svm/kernel.h"
#include <chrono>
#include <vector>
#include <string>
#include <map>


void exit_with_help();
void parse_command_line(svm_param& param, int argc, char **argv, char *features_file_name, char *labels_file_name, char *model_file_name,
                        char * test_features_file_name, char* test_labels_file_name, char* test_predictions_file_name);
void read_q_matrix(svm_param& param, const char* file_name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* Q);
void read_features_matrix(const char* file_name, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* Q);
void read_labels(const char* file_name, Eigen::VectorXd& labels);
void save_model(const char* model_file_name, const Eigen::VectorXd& x, double b);
typedef double (*KernelFunc)(const std::vector<double>& x1, const std::vector<double>& x2, const svm_param& params);
double linearKernel(const std::vector<double>& x1, const std::vector<double>& x2, const svm_param& params);
KernelFunc selectKernelFunction(int kernel_type);
#endif // UTILS_H
