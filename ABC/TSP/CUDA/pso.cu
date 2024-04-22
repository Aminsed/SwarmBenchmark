#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

struct Bee {
    int position[NUM_CITIES];
    double fitness;
};

__device__ double calculateDistance(int* position) {
    double distance = 0.0;
    for (int i = 0; i < NUM_CITIES - 1; i++) {
        int city1 = position[i];
        int city2 = position[i + 1];
        distance += distances[city1][city2];
    }
    int lastCity = position[NUM_CITIES - 1];
    int firstCity = position[0];
    distance += distances[lastCity][firstCity];
    return distance;
}

__device__ void updateBestFitness(Bee* b, int* globalBestPosition, double* globalBestFitness) {
    double fitness = calculateDistance(b->position);
    b->fitness = fitness;
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < NUM_CITIES; i++) {
            globalBestPosition[i] = b->position[i];
        }
    }
}

__global__ void initializeBees(Bee* bees, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_BEES) {
        Bee* b = &bees[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < NUM_CITIES; i++) {
            b->position[i] = i;
        }
        for (int i = NUM_CITIES - 1; i > 0; i--) {
            int j = curand(s) % (i + 1);
            int temp = b->position[i];
            b->position[i] = b->position[j];
            b->position[j] = temp;
        }
        updateBestFitness(b, globalBestPosition, globalBestFitness);
    }
}

__global__ void sendEmployedBees(Bee* bees, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_BEES) {
        Bee* b = &bees[tid];
        curandState* s = &state[tid];
        int newPosition[NUM_CITIES];
        for (int i = 0; i < NUM_CITIES; i++) {
            newPosition[i] = b->position[i];
        }
        int i = curand(s) % NUM_CITIES;
        int j = curand(s) % NUM_CITIES;
        int temp = newPosition[i];
        newPosition[i] = newPosition[j];
        newPosition[j] = temp;
        double newFitness = calculateDistance(newPosition);
        if (newFitness < b->fitness) {
            for (int i = 0; i < NUM_CITIES; i++) {
                b->position[i] = newPosition[i];
            }
            updateBestFitness(b, globalBestPosition, globalBestFitness);
        }
    }
}

__global__ void sendOnlookerBees(Bee* bees, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_BEES) {
        Bee* b = &bees[tid];
        curandState* s = &state[tid];
        double probabilities[NUM_BEES];
        double maxFitness = bees[0].fitness;
        for (int i = 1; i < NUM_BEES; i++) {
            if (bees[i].fitness > maxFitness) {
                maxFitness = bees[i].fitness;
            }
        }
        double fitnessSum = 0.0;
        for (int i = 0; i < NUM_BEES; i++) {
            probabilities[i] = (0.9 * (maxFitness - bees[i].fitness) / maxFitness) + 0.1;
            fitnessSum += probabilities[i];
        }
        double r = curand_uniform_double(s) * fitnessSum;
        double cumulativeProbability = 0.0;
        int selectedBee = -1;
        for (int i = 0; i < NUM_BEES; i++) {
            cumulativeProbability += probabilities[i];
            if (r <= cumulativeProbability) {
                selectedBee = i;
                break;
            }
        }
        if (selectedBee != -1) {
            int newPosition[NUM_CITIES];
            for (int i = 0; i < NUM_CITIES; i++) {
                newPosition[i] = bees[selectedBee].position[i];
            }
            int i = curand(s) % NUM_CITIES;
            int j = curand(s) % NUM_CITIES;
            int temp = newPosition[i];
            newPosition[i] = newPosition[j];
            newPosition[j] = temp;
            double newFitness = calculateDistance(newPosition);
            if (newFitness < bees[selectedBee].fitness) {
                for (int i = 0; i < NUM_CITIES; i++) {
                    bees[selectedBee].position[i] = newPosition[i];
                }
                updateBestFitness(&bees[selectedBee], globalBestPosition, globalBestFitness);
            }
        }
    }
}

__global__ void sendScoutBees(Bee* bees, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_BEES) {
        Bee* b = &bees[tid];
        curandState* s = &state[tid];
        if (curand_uniform_double(s) < SCOUT_BEE_PROBABILITY) {
            for (int i = 0; i < NUM_CITIES; i++) {
                b->position[i] = i;
            }
            for (int i = NUM_CITIES - 1; i > 0; i--) {
                int j = curand(s) % (i + 1);
                int temp = b->position[i];
                b->position[i] = b->position[j];
                b->position[j] = temp;
            }
            updateBestFitness(b, globalBestPosition, globalBestFitness);
        }
    }
}

void runABC(Bee* bees, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_BEES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        sendEmployedBees<<<grid, block>>>(bees, globalBestPosition, globalBestFitness, state);
        cudaDeviceSynchronize();
        sendOnlookerBees<<<grid, block>>>(bees, globalBestPosition, globalBestFitness, state);
        cudaDeviceSynchronize();
        sendScoutBees<<<grid, block>>>(bees, globalBestPosition, globalBestFitness, state);
        cudaDeviceSynchronize();
    }
}

void printResults(int* globalBestPosition, double globalBestFitness, double executionTime) {
    std::cout << "Best Path: ";
    for (int i = 0; i < NUM_CITIES; i++) {
        std::cout << globalBestPosition[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Best Distance: " << globalBestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    Bee* bees;
    int* globalBestPosition;
    double* globalBestFitness;
    curandState* state;
    cudaMalloc(&bees, NUM_BEES * sizeof(Bee));
    cudaMalloc(&globalBestPosition, NUM_CITIES * sizeof(int));
    cudaMalloc(&globalBestFitness, sizeof(double));
    cudaMalloc(&state, NUM_BEES * sizeof(curandState));
    double initialFitness = INFINITY;
    cudaMemcpy(globalBestFitness, &initialFitness, sizeof(double), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    initializeBees<<<(NUM_BEES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(bees, globalBestPosition, globalBestFitness, state);
    cudaDeviceSynchronize();
    runABC(bees, globalBestPosition, globalBestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    int* hostGlobalBestPosition = new int[NUM_CITIES];
    double hostGlobalBestFitness;
    cudaMemcpy(hostGlobalBestPosition, globalBestPosition, NUM_CITIES * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostGlobalBestFitness, globalBestFitness, sizeof(double), cudaMemcpyDeviceToHost);
    printResults(hostGlobalBestPosition, hostGlobalBestFitness, executionTime);
    delete[] hostGlobalBestPosition;
    cudaFree(bees);
    cudaFree(globalBestPosition);
    cudaFree(globalBestFitness);
    cudaFree(state);
    cudaDeviceReset();
    return 0;
}
