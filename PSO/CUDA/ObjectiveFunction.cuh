#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include <cuda_runtime.h>
#include <cmath>

#define DIMENSION 2
#define NUM_PARTICLES 1024
#define MAX_ITERATIONS 1000
#define COGNITIVE_WEIGHT 1.49
#define SOCIAL_WEIGHT 1.49
#define INERTIA_WEIGHT 0.729
#define BLOCK_SIZE 256

class ObjectiveFunction {
public:
    __device__ static double rastrigin(double x, double y) {
        return 20 + (x * x - 10 * cos(2 * M_PI * x)) + (y * y - 10 * cos(2 * M_PI * y));
    }
};

#endif
