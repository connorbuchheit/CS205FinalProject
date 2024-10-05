// serial_icf.h

#ifndef SERIALICF_H
#define SERIALICF_H

#include <vector>
#include <cmath>
#include "svm.h"

void serialICF(const Matrix& Q, int n, int p, Matrix& H);
double calculateTraceDifference(const Matrix& Q, const Matrix& H);
bool checkICFCriterion(const Matrix& Q, const Matrix& H, double epsilon);
#endif // SERIAL_ICF_H