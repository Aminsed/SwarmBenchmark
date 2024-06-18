#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>

struct Salp {
    double position[DIMENSIONS];
    double bestPosition[DIMENSIONS];
    double bestFitness;
};

__device__ void updateBestFitness(Salp* s, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(s->position);
    if (fitness < s->bestFitness) {
        s->bestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            s->bestPosition[i] = s->position[i];
        }
    }
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = s->position[i];
        }
    }
}

__global__ void initializeSalps(Salp* salps, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_SALPS) {
        Salp* s = &salps[tid];
        curandState* st = &state[tid];
        curand_init(clock64(), tid, 0, st);
        for (int i = 0; i < DIMENSIONS; i++) {
            s->position[i] = curand_uniform_double(st) * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND;
            s->bestPosition[i] = s->position[i];
        }
        s->bestFitness = INFINITY;
        updateBestFitness(s, globalBestPosition, globalBestFitness);
    }
}

__global__ void updateSalps(Salp* salps, double* globalBestPosition, double* globalBestFitness, curandState* state, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_SALPS) {
        Salp* s = &salps[tid];
        curandState* st = &state[tid];
        double c1 = 2.0 * exp(-(4.0 * iter / MAX_ITERATIONS) * (4.0 * iter / MAX_ITERATIONS));
        double c2 = curand_uniform_double(st);
        double c3 = curand_uniform_double(st);
        for (int i = 0; i < DIMENSIONS; i++) {
            if (tid == 0) {
                s->position[i] = globalBestPosition[i] + c1 * ((UPPER_BOUND - LOWER_BOUND) * c2 + LOWER_BOUND);
            } else {
                s->position[i] = 0.5 * (s->position[i] + globalBestPosition[i]) + c1 * ((UPPER_BOUND - LOWER_BOUND) * c3 + LOWER_BOUND);
            }
        }
        updateBestFitness(s, globalBestPosition, globalBestFitness);
    }
}

void runSSA(Salp* salps, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_SALPS + block.x - 1) / block.x);
    std::ofstream outputFile("results.txt");
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateSalps<<<grid, block>>>(salps, globalBestPosition, globalBestFitness, state, iter);
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
    Salp* salps;
    double* globalBestPosition;
    double* globalBestFitness;
    curandState* state;
    cudaMalloc(&salps, NUM_SALPS * sizeof(Salp));
    cudaMalloc(&globalBestPosition, DIMENSIONS * sizeof(double));
    cudaMalloc(&globalBestFitness, sizeof(double));
    cudaMalloc(&state, NUM_SALPS * sizeof(curandState));
    double initialFitness = INFINITY;
    cudaMemcpy(globalBestFitness, &initialFitness, sizeof(double), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    initializeSalps<<<(NUM_SALPS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(salps, globalBestPosition, globalBestFitness, state);
    cudaDeviceSynchronize();
    runSSA(salps, globalBestPosition, globalBestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double hostGlobalBestPosition[DIMENSIONS];
    double hostGlobalBestFitness;
    cudaMemcpy(hostGlobalBestPosition, globalBestPosition, DIMENSIONS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostGlobalBestFitness, globalBestFitness, sizeof(double), cudaMemcpyDeviceToHost);
    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);
    cudaFree(salps);
    cudaFree(globalBestPosition);
    cudaFree(globalBestFitness);
    cudaFree(state);
    cudaDeviceReset();
    return 0;
}
