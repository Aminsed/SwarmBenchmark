#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>

struct Whale {
    double position[DIMENSIONS];
    double bestPosition[DIMENSIONS];
    double bestFitness;
};

__device__ void updateBestFitness(Whale* w, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(w->position);
    if (fitness < w->bestFitness) {
        w->bestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            w->bestPosition[i] = w->position[i];
        }
    }
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = w->position[i];
        }
    }
}

__global__ void initializeWhales(Whale* whales, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_WHALES) {
        Whale* w = &whales[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < DIMENSIONS; i++) {
            w->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
            w->bestPosition[i] = w->position[i];
        }
        w->bestFitness = INFINITY;        updateBestFitness(w, globalBestPosition, globalBestFitness);
    }
}

__global__ void updateWhales(Whale* whales, double* globalBestPosition, double* globalBestFitness, curandState* state, int iter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_WHALES) {
        Whale* w = &whales[tid];
        curandState* s = &state[tid];
        double a = 2.0 - (double)iter / MAX_ITERATIONS * 2.0;
        double r1 = curand_uniform_double(s);
        double r2 = curand_uniform_double(s);
        double A = 2.0 * a * r1 - a;
        double C = 2.0 * r2;
        double l = curand_uniform_double(s) * 2.0 - 1.0;
        double p = curand_uniform_double(s);
        for (int i = 0; i < DIMENSIONS; i++) {
            if (p < 0.5) {
                if (fabs(A) < 1.0) {
                    double D = fabs(C * globalBestPosition[i] - w->position[i]);
                    w->position[i] = globalBestPosition[i] - A * D;
                } else {
                    int randomIndex = (int)(curand_uniform_double(s) * NUM_WHALES);
                    double D = fabs(C * whales[randomIndex].position[i] - w->position[i]);
                    w->position[i] = whales[randomIndex].position[i] - A * D;
                }
            } else {
                double D = fabs(globalBestPosition[i] - w->position[i]);
                double b = 1.0; // Constant defining the shape of the logarithmic spiral
                w->position[i] = D * exp(b * l) * cos(2.0 * M_PI * l) + globalBestPosition[i];
            }
        }
        updateBestFitness(w, globalBestPosition, globalBestFitness);
    }
}

void runWOA(Whale* whales, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    std::ofstream outputFile("results.txt");
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_WHALES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateWhales<<<grid, block>>>(whales, globalBestPosition, globalBestFitness, state, iter);
        cudaDeviceSynchronize();
        double hostGlobalBestFitness;
        cudaMemcpy(&hostGlobalBestFitness, globalBestFitness, sizeof(double), cudaMemcpyDeviceToHost);
        outputFile << iter + 1 << ": " << hostGlobalBestFitness << std::endl;
    }    outputFile.close();
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
    Whale* whales;
    double* globalBestPosition;
    double* globalBestFitness;
    curandState* state;
    cudaMalloc(&whales, NUM_WHALES * sizeof(Whale));
    cudaMalloc(&globalBestPosition, DIMENSIONS * sizeof(double));
    cudaMalloc(&globalBestFitness, sizeof(double));
    cudaMalloc(&state, NUM_WHALES * sizeof(curandState));
    double initialFitness = INFINITY;
    cudaMemcpy(globalBestFitness, &initialFitness, sizeof(double), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    initializeWhales<<<(NUM_WHALES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(whales, globalBestPosition, globalBestFitness, state);
    cudaDeviceSynchronize();
    runWOA(whales, globalBestPosition, globalBestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double hostGlobalBestPosition[DIMENSIONS];
    double hostGlobalBestFitness;
    cudaMemcpy(hostGlobalBestPosition, globalBestPosition, DIMENSIONS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostGlobalBestFitness, globalBestFitness, sizeof(double), cudaMemcpyDeviceToHost);
    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);
    cudaFree(whales);
    cudaFree(globalBestPosition);
    cudaFree(globalBestFitness);
    cudaFree(state);
    cudaDeviceReset();
    return 0;
}