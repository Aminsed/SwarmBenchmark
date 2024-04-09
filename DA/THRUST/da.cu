// DragonflyAlgorithm.cu
#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Dragonfly {
    double position[DIMENSIONS];
    double step[DIMENSIONS];
};

__device__ void updateBestFitness(Dragonfly* d, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(d->position);
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = d->position[i];
        }
    }
}

__global__ void initializeDragonflies(Dragonfly* dragonflies, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_DRAGONFLIES) {
        Dragonfly* d = &dragonflies[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < DIMENSIONS; i++) {
            d->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
            d->step[i] = curand_uniform_double(s) * 2.0 - 1.0;
        }
        updateBestFitness(d, globalBestPosition, globalBestFitness);
    }
}

__global__ void updateDragonflies(Dragonfly* dragonflies, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_DRAGONFLIES) {
        Dragonfly* d = &dragonflies[tid];
        curandState* s = &state[tid];
        for (int i = 0; i < DIMENSIONS; i++) {
            double r = curand_uniform_double(s);
            d->step[i] = SEPARATION_WEIGHT * (globalBestPosition[i] - d->position[i]) +
                         ALIGNMENT_WEIGHT * d->step[i] +
                         COHESION_WEIGHT * (globalBestPosition[i] - d->position[i]) +
                         FOOD_ATTRACTION_WEIGHT * (globalBestPosition[i] - d->position[i]) +
                         ENEMY_DISTRACTION_WEIGHT * (curand_uniform_double(s) * 10.0 - 5.0 - d->position[i]);
            d->position[i] += d->step[i];
        }
        updateBestFitness(d, globalBestPosition, globalBestFitness);
    }
}

void runDragonflyAlgorithm(Dragonfly* dragonflies, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_DRAGONFLIES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateDragonflies<<<grid, block>>>(dragonflies, globalBestPosition, globalBestFitness, state);
        cudaDeviceSynchronize();
    }
}

void printResults(double* globalBestPosition, double globalBestFitness, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Global Best Position: " << globalBestPosition[0] << std::endl;
    } else {
        std::cout << "Global Best Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << globalBestPosition[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Global Best Value: " << globalBestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    thrust::device_vector<Dragonfly> dragonflies(NUM_DRAGONFLIES);
    thrust::device_vector<double> globalBestPosition(DIMENSIONS);
    thrust::device_vector<double> globalBestFitness(1);
    curandState* state;
    cudaMalloc(&state, NUM_DRAGONFLIES * sizeof(curandState));

    double initialFitness = INFINITY;
    thrust::copy(&initialFitness, &initialFitness + 1, globalBestFitness.begin());

    auto start = std::chrono::high_resolution_clock::now();
    initializeDragonflies<<<(NUM_DRAGONFLIES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(dragonflies.data()),
        thrust::raw_pointer_cast(globalBestPosition.data()),
        thrust::raw_pointer_cast(globalBestFitness.data()),
        state);
    cudaDeviceSynchronize();

    runDragonflyAlgorithm(
        thrust::raw_pointer_cast(dragonflies.data()),
        thrust::raw_pointer_cast(globalBestPosition.data()),
        thrust::raw_pointer_cast(globalBestFitness.data()),
        state);

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<double> hostGlobalBestPosition = globalBestPosition;
    thrust::host_vector<double> hostGlobalBestFitness = globalBestFitness;

    printResults(thrust::raw_pointer_cast(hostGlobalBestPosition.data()), hostGlobalBestFitness[0], executionTime);

    cudaFree(state);
    return 0;
}
