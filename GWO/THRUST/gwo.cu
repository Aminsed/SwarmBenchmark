#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

struct GreyWolf {
    double position[DIMENSIONS];
    double fitness;
};

__device__ void updateLeaders(GreyWolf* wolf, double* alpha, double* beta, double* delta) {
    if (wolf->fitness < objectiveFunction(alpha)) {
        for (int i = 0; i < DIMENSIONS; i++) {
            delta[i] = beta[i];
            beta[i] = alpha[i];
            alpha[i] = wolf->position[i];
        }
    } else if (wolf->fitness < objectiveFunction(beta)) {
        for (int i = 0; i < DIMENSIONS; i++) {
            delta[i] = beta[i];
            beta[i] = wolf->position[i];
        }
    } else if (wolf->fitness < objectiveFunction(delta)) {
        for (int i = 0; i < DIMENSIONS; i++) {
            delta[i] = wolf->position[i];
        }
    }
}

__global__ void initializeGreyWolves(GreyWolf* wolves, double* alpha, double* beta, double* delta, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_WOLVES) {
        GreyWolf* wolf = &wolves[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < DIMENSIONS; i++) {
            wolf->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
        }
        wolf->fitness = objectiveFunction(wolf->position);
        updateLeaders(wolf, alpha, beta, delta);
    }
}

__global__ void updateGreyWolves(GreyWolf* wolves, double* alpha, double* beta, double* delta, curandState* state, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_WOLVES) {
        GreyWolf* wolf = &wolves[tid];
        curandState* s = &state[tid];
        double a = 2.0 - (double)iter / MAX_ITERATIONS * 2.0;
        for (int i = 0; i < DIMENSIONS; i++) {
            double r1 = curand_uniform_double(s);
            double r2 = curand_uniform_double(s);
            double A1 = 2.0 * a * r1 - a;
            double C1 = 2.0 * r2;
            double D_alpha = abs(C1 * alpha[i] - wolf->position[i]);
            double X1 = alpha[i] - A1 * D_alpha;
            r1 = curand_uniform_double(s);
            r2 = curand_uniform_double(s);
            double A2 = 2.0 * a * r1 - a;
            double C2 = 2.0 * r2;
            double D_beta = abs(C2 * beta[i] - wolf->position[i]);
            double X2 = beta[i] - A2 * D_beta;
            r1 = curand_uniform_double(s);
            r2 = curand_uniform_double(s);
            double A3 = 2.0 * a * r1 - a;
            double C3 = 2.0 * r2;
            double D_delta = abs(C3 * delta[i] - wolf->position[i]);
            double X3 = delta[i] - A3 * D_delta;
            wolf->position[i] = (X1 + X2 + X3) / 3.0;
        }
        wolf->fitness = objectiveFunction(wolf->position);
        updateLeaders(wolf, alpha, beta, delta);
    }
}

void runGWO(GreyWolf* wolves, double* alpha, double* beta, double* delta, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_WOLVES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateGreyWolves<<<grid, block>>>(wolves, alpha, beta, delta, state, iter);
        cudaDeviceSynchronize();
    }
}

__global__ void calculateAlphaFitness(double* alpha, double* alphaFitness) {
    if (threadIdx.x == 0) {
        *alphaFitness = objectiveFunction(alpha);
    }
}

void printResults(double* alpha, double alphaFitness, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Best Position: " << alpha[0] << std::endl;
    } else {
        std::cout << "Best Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << alpha[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Fitness: " << alphaFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    thrust::device_vector<GreyWolf> d_wolves(NUM_WOLVES);
    thrust::device_vector<double> d_alpha(DIMENSIONS);
    thrust::device_vector<double> d_beta(DIMENSIONS);
    thrust::device_vector<double> d_delta(DIMENSIONS);
    thrust::device_vector<curandState> d_state(NUM_WOLVES);
    thrust::device_vector<double> d_alphaFitness(1);

    auto start = std::chrono::high_resolution_clock::now();

    initializeGreyWolves<<<(NUM_WOLVES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_wolves.data()),
        thrust::raw_pointer_cast(d_alpha.data()),
        thrust::raw_pointer_cast(d_beta.data()),
        thrust::raw_pointer_cast(d_delta.data()),
        thrust::raw_pointer_cast(d_state.data())
    );
    cudaDeviceSynchronize();

    runGWO(
        thrust::raw_pointer_cast(d_wolves.data()),
        thrust::raw_pointer_cast(d_alpha.data()),
        thrust::raw_pointer_cast(d_beta.data()),
        thrust::raw_pointer_cast(d_delta.data()),
        thrust::raw_pointer_cast(d_state.data())
    );

    calculateAlphaFitness<<<1, 1>>>(
        thrust::raw_pointer_cast(d_alpha.data()),
        thrust::raw_pointer_cast(d_alphaFitness.data())
    );
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double hostAlpha[DIMENSIONS];
    thrust::copy(d_alpha.begin(), d_alpha.end(), hostAlpha);
    double alphaFitness = d_alphaFitness[0];

    printResults(hostAlpha, alphaFitness, executionTime);

    return 0;
}
