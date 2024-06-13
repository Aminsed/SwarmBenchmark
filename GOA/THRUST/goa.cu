// GrasshopperOptimizationAlgorithm.cu
#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <fstream>

struct Grasshopper {
    double position[DIMENSIONS];
};

__device__ void updateBestFitness(Grasshopper* g, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(g->position);
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = g->position[i];
        }
    }
}

__global__ void initializeGrasshoppers(Grasshopper* grasshoppers, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_GRASSHOPPERS) {
        Grasshopper* g = &grasshoppers[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < DIMENSIONS; i++) {
            g->position[i] = curand_uniform_double(s) * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND;
        }
        updateBestFitness(g, globalBestPosition, globalBestFitness);
    }
}

__global__ void updateGrasshoppers(Grasshopper* grasshoppers, double* globalBestPosition, double* globalBestFitness, curandState* state, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_GRASSHOPPERS) {
        Grasshopper* g = &grasshoppers[tid];
        curandState* s = &state[tid];
        double c = 0.00001;
        double cMax = 1.0;
        double cMin = 0.00001;
        double l = (cMax - cMin) * (MAX_ITERATIONS - iter) / MAX_ITERATIONS + cMin;
        for (int i = 0; i < DIMENSIONS; i++) {
            double socialInteraction = 0.0;
            for (int j = 0; j < NUM_GRASSHOPPERS; j++) {
                if (j != tid) {
                    double distance = fabs(g->position[i] - grasshoppers[j].position[i]);
                    double r = curand_uniform_double(s);
                    double si = (0.5 + 0.5 * r) * (globalBestPosition[i] - l * distance);
                    socialInteraction += si;
                }
            }
            double xi = c * socialInteraction;
            double r = curand_uniform_double(s);
            double levy = pow(r, -1.0 / LEVY_EXPONENT);
            double newPosition = g->position[i] + xi * levy;
            g->position[i] = fmax(LOWER_BOUND, fmin(newPosition, UPPER_BOUND));
        }
        updateBestFitness(g, globalBestPosition, globalBestFitness);
    }
}

void runGOA(thrust::device_vector<Grasshopper>& grasshoppers, thrust::device_vector<double>& globalBestPosition, thrust::device_vector<double>& globalBestFitness, thrust::device_vector<curandState>& state) {
    std::ofstream outputFile("results.txt");
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_GRASSHOPPERS + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateGrasshoppers<<<grid, block>>>(thrust::raw_pointer_cast(grasshoppers.data()), thrust::raw_pointer_cast(globalBestPosition.data()), thrust::raw_pointer_cast(globalBestFitness.data()), thrust::raw_pointer_cast(state.data()), iter);
        cudaDeviceSynchronize();
        double hostGlobalBestFitness = globalBestFitness[0];
        outputFile << iter + 1 << ": " << hostGlobalBestFitness << std::endl;
    }
    outputFile.close();
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
    thrust::device_vector<Grasshopper> grasshoppers(NUM_GRASSHOPPERS);
    thrust::device_vector<double> globalBestPosition(DIMENSIONS);
    thrust::device_vector<double> globalBestFitness(1);
    thrust::device_vector<curandState> state(NUM_GRASSHOPPERS);
    double initialFitness = INFINITY;
    thrust::copy(&initialFitness, &initialFitness + 1, globalBestFitness.begin());
    auto start = std::chrono::high_resolution_clock::now();
    initializeGrasshoppers<<<(NUM_GRASSHOPPERS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(grasshoppers.data()), thrust::raw_pointer_cast(globalBestPosition.data()), thrust::raw_pointer_cast(globalBestFitness.data()), thrust::raw_pointer_cast(state.data()));
    cudaDeviceSynchronize();
    runGOA(grasshoppers, globalBestPosition, globalBestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    thrust::host_vector<double> hostGlobalBestPosition(DIMENSIONS);
    thrust::copy(globalBestPosition.begin(), globalBestPosition.end(), hostGlobalBestPosition.begin());
    double hostGlobalBestFitness = globalBestFitness[0];
    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);
    return 0;
}
