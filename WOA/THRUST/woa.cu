#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

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
        w->bestFitness = INFINITY;
        updateBestFitness(w, globalBestPosition, globalBestFitness);
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
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_WHALES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateWhales<<<grid, block>>>(whales, globalBestPosition, globalBestFitness, state, iter);
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
    thrust::device_vector<Whale> d_whales(NUM_WHALES);
    thrust::device_vector<double> d_globalBestPosition(DIMENSIONS);
    thrust::device_vector<double> d_globalBestFitness(1);
    thrust::device_vector<curandState> d_state(NUM_WHALES);
    double initialFitness = INFINITY;
    d_globalBestFitness[0] = initialFitness;
    auto start = std::chrono::high_resolution_clock::now();
    initializeWhales<<<(NUM_WHALES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_whales.data()),
        thrust::raw_pointer_cast(d_globalBestPosition.data()),
        thrust::raw_pointer_cast(d_globalBestFitness.data()),
        thrust::raw_pointer_cast(d_state.data())
    );
    cudaDeviceSynchronize();
    runWOA(
        thrust::raw_pointer_cast(d_whales.data()),
        thrust::raw_pointer_cast(d_globalBestPosition.data()),
        thrust::raw_pointer_cast(d_globalBestFitness.data()),
        thrust::raw_pointer_cast(d_state.data())
    );
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double hostGlobalBestPosition[DIMENSIONS];
    double hostGlobalBestFitness = d_globalBestFitness[0];
    thrust::copy(d_globalBestPosition.begin(), d_globalBestPosition.end(), hostGlobalBestPosition);
    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);
    return 0;
}
