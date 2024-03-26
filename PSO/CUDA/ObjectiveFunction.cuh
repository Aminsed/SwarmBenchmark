#pragma once

#include <cmath>

#define DIMENSION 2
#define NUM_PARTICLES 1024
#define MAX_ITERATIONS 1000
#define COGNITIVE_WEIGHT 1.49f
#define SOCIAL_WEIGHT 1.49f
#define INERTIA_WEIGHT 0.729f
#define BLOCK_SIZE 256

// rastrigin
__device__ double objectiveFunction(double* position, int dimension) {
    double sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += std::pow(position[i], 2) - 10.0 * std::cos(2.0 * M_PI * position[i]);
    }
    return sum;
}
