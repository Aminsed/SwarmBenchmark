#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <fstream>

struct GreyWolf {
    double position[DIMENSIONS];
    double fitness;
};

__device__ void updateLeaders(GreyWolf* wolf, double* alpha, double* beta, double* delta) {
    if (wolf->fitness < alpha[DIMENSIONS]) {
        for (int i = 0; i < DIMENSIONS; i++) {
            alpha[i] = wolf->position[i];
        }
        alpha[DIMENSIONS] = wolf->fitness;
    }
    if (wolf->fitness < beta[DIMENSIONS] && wolf->fitness > alpha[DIMENSIONS]) {
        for (int i = 0; i < DIMENSIONS; i++) {
            beta[i] = wolf->position[i];
        }
        beta[DIMENSIONS] = wolf->fitness;
    }
    if (wolf->fitness < delta[DIMENSIONS] && wolf->fitness > beta[DIMENSIONS]) {
        for (int i = 0; i < DIMENSIONS; i++) {
            delta[i] = wolf->position[i];
        }
        delta[DIMENSIONS] = wolf->fitness;
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
    std::ofstream outputFile("results.txt");
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateGreyWolves<<<grid, block>>>(wolves, alpha, beta, delta, state, iter);
        cudaDeviceSynchronize();
        double bestFitness;
        cudaMemcpy(&bestFitness, &alpha[DIMENSIONS], sizeof(double), cudaMemcpyDeviceToHost);
        outputFile << iter + 1 << ": " << bestFitness << std::endl;
    }
    outputFile.close();
}

void printResults(double* alpha, double executionTime) {
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
    std::cout << "Best Fitness: " << alpha[DIMENSIONS] << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    thrust::device_vector<GreyWolf> d_wolves(NUM_WOLVES);
    thrust::device_vector<double> d_alpha(DIMENSIONS + 1);
    thrust::device_vector<double> d_beta(DIMENSIONS + 1);
    thrust::device_vector<double> d_delta(DIMENSIONS + 1);
    thrust::device_vector<curandState> d_state(NUM_WOLVES);

    double initialFitness = INFINITY;
    thrust::fill(d_alpha.begin() + DIMENSIONS, d_alpha.end(), initialFitness);
    thrust::fill(d_beta.begin() + DIMENSIONS, d_beta.end(), initialFitness);
    thrust::fill(d_delta.begin() + DIMENSIONS, d_delta.end(), initialFitness);

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

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double hostAlpha[DIMENSIONS + 1];
    cudaMemcpy(hostAlpha, thrust::raw_pointer_cast(d_alpha.data()), (DIMENSIONS + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    printResults(hostAlpha, executionTime);

    return 0;
}