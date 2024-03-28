#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

struct Firefly {
    double position[DIMENSIONS];
    double brightness;
};

__device__ double attractiveness(double distance) {
    return BETA0 * exp(-GAMMA * distance * distance);
}

__global__ void initializeFireflies(Firefly* fireflies, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_FIREFLIES) {
        Firefly* f = &fireflies[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);

        for (int i = 0; i < DIMENSIONS; i++) {
            f->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
        }
        f->brightness = objectiveFunction(f->position);

        if (f->brightness < *globalBestFitness) {
            *globalBestFitness = f->brightness;
            for (int i = 0; i < DIMENSIONS; i++) {
                globalBestPosition[i] = f->position[i];
            }
        }
    }
}

__global__ void updateFireflies(Firefly* fireflies, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_FIREFLIES) {
        Firefly* f = &fireflies[tid];
        curandState* s = &state[tid];

        for (int i = 0; i < NUM_FIREFLIES; i++) {
            if (i != tid) {
                Firefly* otherFirefly = &fireflies[i];
                double distance = 0.0;
                for (int j = 0; j < DIMENSIONS; j++) {
                    double diff = f->position[j] - otherFirefly->position[j];
                    distance += diff * diff;
                }
                distance = sqrt(distance);

                if (otherFirefly->brightness > f->brightness) {
                    double beta = attractiveness(distance);
                    for (int j = 0; j < DIMENSIONS; j++) {
                        double r = curand_uniform_double(s);
                        f->position[j] += beta * (otherFirefly->position[j] - f->position[j]) + ALPHA * (r - 0.5);
                    }
                }
            }
        }

        f->brightness = objectiveFunction(f->position);

        if (f->brightness < *globalBestFitness) {
            *globalBestFitness = f->brightness;
            for (int i = 0; i < DIMENSIONS; i++) {
                globalBestPosition[i] = f->position[i];
            }
        }
    }
}

void runFA(Firefly* fireflies, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_FIREFLIES + block.x - 1) / block.x);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateFireflies<<<grid, block>>>(fireflies, globalBestPosition, globalBestFitness, state);
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
    thrust::device_vector<Firefly> fireflies(NUM_FIREFLIES);
    thrust::device_vector<double> globalBestPosition(DIMENSIONS);
    thrust::device_vector<double> globalBestFitness(1);
    thrust::device_vector<curandState> state(NUM_FIREFLIES);

    double initialFitness = INFINITY;
    globalBestFitness[0] = initialFitness;

    auto start = std::chrono::high_resolution_clock::now();

    initializeFireflies<<<(NUM_FIREFLIES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(fireflies.data()),
        thrust::raw_pointer_cast(globalBestPosition.data()),
        thrust::raw_pointer_cast(globalBestFitness.data()),
        thrust::raw_pointer_cast(state.data())
    );
    cudaDeviceSynchronize();

    runFA(
        thrust::raw_pointer_cast(fireflies.data()),
        thrust::raw_pointer_cast(globalBestPosition.data()),
        thrust::raw_pointer_cast(globalBestFitness.data()),
        thrust::raw_pointer_cast(state.data())
    );

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double hostGlobalBestPosition[DIMENSIONS];
    double hostGlobalBestFitness = globalBestFitness[0];
    thrust::copy(globalBestPosition.begin(), globalBestPosition.end(), hostGlobalBestPosition);

    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);

    return 0;
}
