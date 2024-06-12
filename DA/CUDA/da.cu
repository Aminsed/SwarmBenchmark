#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>


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
        double separation[DIMENSIONS] = {0};
        double alignment[DIMENSIONS] = {0};
        double cohesion[DIMENSIONS] = {0};
        for (int j = 0; j < NUM_DRAGONFLIES; j++) {
            if (tid != j) {
                Dragonfly* neighbor = &dragonflies[j];
                for (int k = 0; k < DIMENSIONS; k++) {
                    double diff = d->position[k] - neighbor->position[k];
                    separation[k] += diff;
                    alignment[k] += neighbor->step[k];
                    cohesion[k] += neighbor->position[k];                }
            }
        }
        for (int i = 0; i < DIMENSIONS; i++) {
            separation[i] /= (NUM_DRAGONFLIES - 1);
            alignment[i] /= (NUM_DRAGONFLIES - 1);
            cohesion[i] /= (NUM_DRAGONFLIES - 1);

            double r = curand_uniform_double(s);
            d->step[i] = SEPARATION_WEIGHT * separation[i] +
                         ALIGNMENT_WEIGHT * alignment[i] +
                         COHESION_WEIGHT * (cohesion[i] - d->position[i]) +
                         FOOD_ATTRACTION_WEIGHT * (globalBestPosition[i] - d->position[i]) +
                         ENEMY_DISTRACTION_WEIGHT * (curand_uniform_double(s) - 0.5);
        }
        for (int i = 0; i < DIMENSIONS; i++) {
            d->position[i] += d->step[i];
        }
        updateBestFitness(d, globalBestPosition, globalBestFitness);
    }
}

void runDragonflyAlgorithm(Dragonfly* dragonflies, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    std::ofstream outputFile("results.txt");
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_DRAGONFLIES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateDragonflies<<<grid, block>>>(dragonflies, globalBestPosition, globalBestFitness, state);
        cudaDeviceSynchronize();
        double hostGlobalBestFitness;
        cudaMemcpy(&hostGlobalBestFitness, globalBestFitness, sizeof(double), cudaMemcpyDeviceToHost);
        outputFile << iter + 1 << ": " << hostGlobalBestFitness << std::endl;
    }
    outputFile.close();
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
    Dragonfly* dragonflies;
    double* globalBestPosition;
    double* globalBestFitness;
    curandState* state;
    cudaMalloc(&dragonflies, NUM_DRAGONFLIES * sizeof(Dragonfly));
    cudaMalloc(&globalBestPosition, DIMENSIONS * sizeof(double));
    cudaMalloc(&globalBestFitness, sizeof(double));
    cudaMalloc(&state, NUM_DRAGONFLIES * sizeof(curandState));
    double initialFitness = INFINITY;
    cudaMemcpy(globalBestFitness, &initialFitness, sizeof(double), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    initializeDragonflies<<<(NUM_DRAGONFLIES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dragonflies, globalBestPosition, globalBestFitness, state);
    cudaDeviceSynchronize();
    runDragonflyAlgorithm(dragonflies, globalBestPosition, globalBestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double hostGlobalBestPosition[DIMENSIONS];
    double hostGlobalBestFitness;
    cudaMemcpy(hostGlobalBestPosition, globalBestPosition, DIMENSIONS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostGlobalBestFitness, globalBestFitness, sizeof(double), cudaMemcpyDeviceToHost);
    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);
    cudaFree(dragonflies);
    cudaFree(globalBestPosition);
    cudaFree(globalBestFitness);
    cudaFree(state);
    cudaDeviceReset();
    return 0;
}
