#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

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
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateSalps<<<grid, block>>>(salps, globalBestPosition, globalBestFitness, state, iter);
        cudaDeviceSynchronize();
    }
}

void printResults(thrust::host_vector<double>& globalBestPosition, double globalBestFitness, double executionTime) {
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
    thrust::device_vector<Salp> salps(NUM_SALPS);
    thrust::device_vector<double> globalBestPosition(DIMENSIONS);
    thrust::device_vector<double> globalBestFitness(1);
    thrust::device_vector<curandState> state(NUM_SALPS);
    double initialFitness = INFINITY;
    thrust::copy(&initialFitness, &initialFitness + 1, globalBestFitness.begin());
    auto start = std::chrono::high_resolution_clock::now();
    initializeSalps<<<(NUM_SALPS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(salps.data()), thrust::raw_pointer_cast(globalBestPosition.data()), thrust::raw_pointer_cast(globalBestFitness.data()), thrust::raw_pointer_cast(state.data()));
    cudaDeviceSynchronize();
    runSSA(thrust::raw_pointer_cast(salps.data()), thrust::raw_pointer_cast(globalBestPosition.data()), thrust::raw_pointer_cast(globalBestFitness.data()), thrust::raw_pointer_cast(state.data()));
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    thrust::host_vector<double> hostGlobalBestPosition = globalBestPosition;
    double hostGlobalBestFitness = globalBestFitness[0];
    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);
    return 0;
}
