#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <fstream>

struct Ant {
    double position[DIMENSIONS];
    double fitness;
};

__device__ void updatePheromone(double* pheromone, double* bestPosition, double bestFitness) {
    for (int i = 0; i < DIMENSIONS; i++) {
        pheromone[i] += Q / (bestFitness + 1e-10);
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
            pheromone[i] = 1.0;
        }
        a->fitness = objectiveFunction(a->position);
    }
}

__global__ void updateAnts(Ant* ants, double* pheromone, double* bestPosition, double* bestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double sharedBestPosition[DIMENSIONS];
    __shared__ double sharedBestFitness;

    if (threadIdx.x == 0) {
        sharedBestFitness = *bestFitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            sharedBestPosition[i] = bestPosition[i];
        }
    }
    __syncthreads();

    if (tid < NUM_ANTS) {
        Ant* a = &ants[tid];
        curandState* s = &state[tid];
        for (int i = 0; i < DIMENSIONS; i++) {
            double r = curand_uniform_double(s);
            if (r < PHEROMONE_WEIGHT) {
                a->position[i] = sharedBestPosition[i] + (curand_uniform_double(s) * 2.0 - 1.0);
            } else {
                a->position[i] += curand_uniform_double(s) * 2.0 - 1.0;
            }
        }
        a->fitness = objectiveFunction(a->position);
        
        // Update best fitness and position using shared memory
        if (a->fitness < sharedBestFitness) {
            sharedBestFitness = a->fitness;
            for (int i = 0; i < DIMENSIONS; i++) {
                sharedBestPosition[i] = a->position[i];
            }
        }
    }
    __syncthreads();

    // Update global best fitness and position
    if (threadIdx.x == 0) {
        if (sharedBestFitness < *bestFitness) {
            *bestFitness = sharedBestFitness;
            for (int i = 0; i < DIMENSIONS; i++) {
                bestPosition[i] = sharedBestPosition[i];
            }
        }
    }
}

void runACO(Ant* ants, double* pheromone, double* bestPosition, double* bestFitness, curandState* state) {
    std::ofstream outputFile("results.txt");
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_ANTS + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateAnts<<<grid, block>>>(ants, pheromone, bestPosition, bestFitness, state);
        cudaDeviceSynchronize();
        double hostBestFitness;
        cudaMemcpy(&hostBestFitness, bestFitness, sizeof(double), cudaMemcpyDeviceToHost);
        outputFile << iter + 1 << ": " << hostBestFitness << std::endl;
    }
    outputFile.close();
}

void printResults(thrust::host_vector<double>& bestPosition, double bestFitness, double executionTime) {
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
    thrust::device_vector<Ant> ants(NUM_ANTS);
    thrust::device_vector<double> pheromone(DIMENSIONS);
    thrust::device_vector<double> bestPosition(DIMENSIONS);
    thrust::device_vector<double> bestFitness(1);
    thrust::device_vector<curandState> state(NUM_ANTS);

    double initialFitness = INFINITY;
    thrust::copy(&initialFitness, &initialFitness + 1, bestFitness.begin());

    auto start = std::chrono::high_resolution_clock::now();
    initializeAnts<<<(NUM_ANTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(ants.data()), thrust::raw_pointer_cast(pheromone.data()), thrust::raw_pointer_cast(state.data()));
    cudaDeviceSynchronize();
    runACO(thrust::raw_pointer_cast(ants.data()), thrust::raw_pointer_cast(pheromone.data()), thrust::raw_pointer_cast(bestPosition.data()), thrust::raw_pointer_cast(bestFitness.data()), thrust::raw_pointer_cast(state.data()));
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<double> hostBestPosition = bestPosition;
    double hostBestFitness = bestFitness[0];

    printResults(hostBestPosition, hostBestFitness, executionTime);

    return 0;
}