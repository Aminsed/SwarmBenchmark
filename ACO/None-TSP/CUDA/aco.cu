#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
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

__global__ void initializeAnts(Ant* ants, double* pheromone, curandState* state, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_ANTS) {
        curand_init(seed, tid, 0, &state[tid]);
        Ant* a = &ants[tid];
        for (int i = 0; i < DIMENSIONS; i++) {
            a->position[i] = curand_uniform_double(&state[tid]) * 10.0 - 5.0;
            pheromone[i] = 1.0;
        }
        a->fitness = objectiveFunction(a->position);
    }
}

__global__ void updateAnts(Ant* ants, double* pheromone, double* bestPosition, double* bestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_ANTS) {
        Ant* a = &ants[tid];
        for (int i = 0; i < DIMENSIONS; i++) {
            double r = curand_uniform_double(&state[tid]);
            if (r < PHEROMONE_WEIGHT) {
                a->position[i] = bestPosition[i] + (curand_uniform_double(&state[tid]) * 2.0 - 1.0);
            } else {
                a->position[i] += curand_uniform_double(&state[tid]) * 2.0 - 1.0;
            }
        }
        a->fitness = objectiveFunction(a->position);
        
        // Update best fitness and position
        if (a->fitness < *bestFitness) {
            *bestFitness = a->fitness;
            for (int i = 0; i < DIMENSIONS; i++) {
                bestPosition[i] = a->position[i];
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
    
    unsigned long long seed = time(NULL);
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_ANTS + block.x - 1) / block.x);
    
    initializeAnts<<<grid, block>>>(ants, pheromone, state, seed);
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
