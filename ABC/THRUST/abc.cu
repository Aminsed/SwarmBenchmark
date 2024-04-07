#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <chrono>
#include <iostream>
#include <iomanip>

struct FoodSource {
    double position[DIMENSIONS];
    double fitness;
    int trialCount;
};

__device__ void updateFitness(FoodSource* fs) {
    fs->fitness = objectiveFunction(fs->position);
}

__global__ void initializeFoodSources(FoodSource* foodSources, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_FOOD_SOURCES) {
        FoodSource* fs = &foodSources[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < DIMENSIONS; i++) {
            fs->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
        }
        updateFitness(fs);
        fs->trialCount = 0;
    }
}

__global__ void sendEmployedBees(FoodSource* foodSources, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_FOOD_SOURCES) {
        FoodSource* fs = &foodSources[tid];
        curandState* s = &state[tid];
        int j = curand_uniform(s) * DIMENSIONS;
        double phi = curand_uniform_double(s) * 2.0 - 1.0;
        FoodSource newFs = *fs;
        newFs.position[j] += phi * (newFs.position[j] - foodSources[(int)(curand_uniform(s) * NUM_FOOD_SOURCES)].position[j]);
        updateFitness(&newFs);
        if (newFs.fitness < fs->fitness) {
            *fs = newFs;
            fs->trialCount = 0;
        } else {
            fs->trialCount++;
        }
    }
}

__global__ void sendOnlookerBees(FoodSource* foodSources, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_FOOD_SOURCES) {
        curandState* s = &state[tid];
        double probabilities[NUM_FOOD_SOURCES];
        double maxFitness = foodSources[0].fitness;
        for (int i = 1; i < NUM_FOOD_SOURCES; i++) {
            if (foodSources[i].fitness > maxFitness) {
                maxFitness = foodSources[i].fitness;
            }
        }
        double fitnessSum = 0.0;
        for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
            probabilities[i] = (0.9 * (foodSources[i].fitness / maxFitness)) + 0.1;
            fitnessSum += probabilities[i];
        }
        double r = curand_uniform_double(s) * fitnessSum;
        double cumulativeProbability = 0.0;
        int selectedIndex = 0;
        for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
            cumulativeProbability += probabilities[i];
            if (r <= cumulativeProbability) {
                selectedIndex = i;
                break;
            }
        }
        FoodSource* fs = &foodSources[selectedIndex];
        int j = curand_uniform(s) * DIMENSIONS;
        double phi = curand_uniform_double(s) * 2.0 - 1.0;
        FoodSource newFs = *fs;
        newFs.position[j] += phi * (newFs.position[j] - foodSources[(int)(curand_uniform(s) * NUM_FOOD_SOURCES)].position[j]);
        updateFitness(&newFs);
        if (newFs.fitness < fs->fitness) {
            *fs = newFs;
            fs->trialCount = 0;
        } else {
            fs->trialCount++;
        }
    }
}

__global__ void sendScoutBees(FoodSource* foodSources, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_FOOD_SOURCES) {
        FoodSource* fs = &foodSources[tid];
        if (fs->trialCount >= LIMIT) {
            curandState* s = &state[tid];
            for (int i = 0; i < DIMENSIONS; i++) {
                fs->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
            }
            updateFitness(fs);
            fs->trialCount = 0;
        }
    }
}

void runABC(thrust::device_vector<FoodSource>& d_foodSources, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_FOOD_SOURCES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        sendEmployedBees<<<grid, block>>>(thrust::raw_pointer_cast(d_foodSources.data()), state);
        cudaDeviceSynchronize();
        sendOnlookerBees<<<grid, block>>>(thrust::raw_pointer_cast(d_foodSources.data()), state);
        cudaDeviceSynchronize();
        sendScoutBees<<<grid, block>>>(thrust::raw_pointer_cast(d_foodSources.data()), state);
        cudaDeviceSynchronize();
    }
}

void printResults(FoodSource* foodSources, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    FoodSource bestFoodSource = foodSources[0];
    for (int i = 1; i < NUM_FOOD_SOURCES; i++) {
        if (foodSources[i].fitness < bestFoodSource.fitness) {
            bestFoodSource = foodSources[i];
        }
    }
    if (DIMENSIONS == 1) {
        std::cout << "Best Food Source Position: " << bestFoodSource.position[0] << std::endl;
    } else {
        std::cout << "Best Food Source Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << bestFoodSource.position[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Food Source Fitness: " << bestFoodSource.fitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    thrust::device_vector<FoodSource> d_foodSources(NUM_FOOD_SOURCES);
    curandState* state;
    cudaMalloc(&state, NUM_FOOD_SOURCES * sizeof(curandState));
    auto start = std::chrono::high_resolution_clock::now();
    initializeFoodSources<<<(NUM_FOOD_SOURCES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_foodSources.data()), state);
    cudaDeviceSynchronize();
    runABC(d_foodSources, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    thrust::host_vector<FoodSource> h_foodSources = d_foodSources;
    printResults(thrust::raw_pointer_cast(h_foodSources.data()), executionTime);
    cudaFree(state);
    return 0;
}
