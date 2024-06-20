#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define NUM_MOTHS 1024
#define MAX_ITERATIONS 1000


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
    double sum = 0.0;
    for (int i = 0; i < DIMENSIONS - 1; i++) {
        double xi = position[i];
        double xi_plus_1 = position[i + 1];
        double numerator = pow(cos(sin(fabs(xi * xi - xi_plus_1 * xi_plus_1))), 2) - 0.5;
        double denominator = pow(1.0 + 0.001 * (xi * xi + xi_plus_1 * xi_plus_1), 2);
        sum += 0.5 + numerator / denominator;
    }
    return sum;
}
*/


#endif
