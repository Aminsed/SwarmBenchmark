#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define NUM_SALPS 1024
#define MAX_ITERATIONS 1000
#define LOWER_BOUND -5.0
#define UPPER_BOUND 5.0

/*
// Rosenbrock function
#define DIMENSIONS 1 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double x = position[0];
    return 100.0 * (x * x - x) * (x * x - x) + (1.0 - x) * (1.0 - x);
}
*/

// Rastrigin function
#define DIMENSIONS 2 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double sum = 0.0;
    for (int i = 0; i < DIMENSIONS; i++) {
        double xi = position[i];
        sum += (xi * xi - 10.0 * cos(2.0 * M_PI * xi));
    }
    return 20.0 + sum;
}

/*
// Griewank function
#define DIMENSIONS 3 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double x = position[0];
    double y = position[1];
    double z = position[2];
    return 1.0 + (x * x + y * y + z * z) / 4000.0 - cos(x) * cos(y / sqrt(2.0)) * cos(z / sqrt(3.0));
}
*/

/*
// Schaffer function N.4
#define DIMENSIONS 4 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double x = position[0];
    double y = position[1];
    double numerator = pow(cos(sin(fabs(x * x - y * y))), 2) - 0.5;
    double denominator = pow(1.0 + 0.001 * (x * x + y * y), 2);
    return 0.5 + numerator / denominator;
}
*/

#endif
