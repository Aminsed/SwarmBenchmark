#ifndef OBJECTIVEFUNCTION_HPP
#define OBJECTIVEFUNCTION_HPP

#include <cmath>

const int DIMENSIONS = 2;
const int NUM_MOTHS = 1024;
const int MAX_ITERATIONS = 1000;

// Sphere Function
/*
double objectiveFunction(double* position) {
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += position[i] * position[i];
    }
    return sum;
}
*/

// Rosenbrock Function
/*
double objectiveFunction(double* position) {
    double sum = 0;
    for (int i = 0; i < DIMENSIONS - 1; i++) {
        double xi = position[i];
        double xnext = position[i + 1];
        double new_sum = 100 * (xnext - xi * xi) * (xnext - xi * xi) + (xi - 1) * (xi - 1);
        sum += new_sum;
    }
    return sum;
}
*/

// Rastrigin Function
double objectiveFunction(double* position) {
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += (position[i] * position[i] - 10 * cos(2 * M_PI * position[i]) + 10);
    }
    return sum;
}

// Griewank Function
/*
double objectiveFunction(double* position) {
    double sum = 0;
    double product = 1;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum += (position[i] * position[i]) / 4000.0;
        product *= cos(position[i] / sqrt(i + 1));
    }
    return sum - product + 1;
}
*/

// Ackley Function
/*
double objectiveFunction(double* position) {
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        sum1 += position[i] * position[i];
        sum2 += cos(2 * M_PI * position[i]);
    }
    return -20 * exp(-0.2 * sqrt(sum1 / DIMENSIONS)) - exp(sum2 / DIMENSIONS) + 20 + exp(1);
}
*/

#endif // OBJECTIVEFUNCTION_HPP
