#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>

struct Ant {
    double position[DIMENSIONS];
    double fitness;
};

__device__ void updatePheromone(double* pheromone, double* bestPosition, double bestFitness) {
    for (int i = 0; i < DIMENSIONS; i++) {
        pheromone[i] += Q / bestFitness;
    }
}

__global__ void initializeAnts(Ant* ants, double* pheromone, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_ANTS) {
        Ant* a = &ants[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < DIMENSIONS; i++) {
            a->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
        }
        a->fitness = objectiveFunction(a->position);
    }
}

__global__ void updateAnts(Ant* ants, double* pheromone, double* bestPosition, double* bestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_ANTS) {
        Ant* a = &ants[tid];
        curandState* s = &state[tid];
        for (int i = 0; i < DIMENSIONS; i++) {
            double r = curand_uniform_double(s);
            if (r < PHEROMONE_WEIGHT) {
                a->position[i] = bestPosition[i];
            } else {
                a->position[i] += curand_uniform_double(s) * 2.0 - 1.0;
            }
        }
        a->fitness = objectiveFunction(a->position);
        if (a->fitness < *bestFitness) {
            *bestFitness = a->fitness;
            for (int i = 0; i < DIMENSIONS; i++) {
                bestPosition[i] = a->position[i];
            }
            updatePheromone(pheromone, bestPosition, *bestFitness);
        }
    }
}

void runACO(Ant* ants, double* pheromone, double* bestPosition, double* bestFitness, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_ANTS + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateAnts<<<grid, block>>>(ants, pheromone, bestPosition, bestFitness, state);
        cudaDeviceSynchronize();
    }
}

void printResults(double* bestPosition, double bestFitness, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Best Position: " << bestPosition[0] << std::endl;
    } else {
        std::cout << "Best Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << bestPosition[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Value: " << bestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    Ant* ants;
    double* pheromone;
    double* bestPosition;
    double* bestFitness;
    curandState* state;
    cudaMalloc(&ants, NUM_ANTS * sizeof(Ant));
    cudaMalloc(&pheromone, DIMENSIONS * sizeof(double));
    cudaMalloc(&bestPosition, DIMENSIONS * sizeof(double));
    cudaMalloc(&bestFitness, sizeof(double));
    cudaMalloc(&state, NUM_ANTS * sizeof(curandState));
    double initialFitness = INFINITY;
    cudaMemcpy(bestFitness, &initialFitness, sizeof(double), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    initializeAnts<<<(NUM_ANTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(ants, pheromone, state);
    cudaDeviceSynchronize();
    runACO(ants, pheromone, bestPosition, bestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double hostBestPosition[DIMENSIONS];
    double hostBestFitness;
    cudaMemcpy(hostBestPosition, bestPosition, DIMENSIONS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostBestFitness, bestFitness, sizeof(double), cudaMemcpyDeviceToHost);
    printResults(hostBestPosition, hostBestFitness, executionTime);
    cudaFree(ants);
    cudaFree(pheromone);
    cudaFree(bestPosition);
    cudaFree(bestFitness);
    cudaFree(state);
    cudaDeviceReset();
    return 0;
}
