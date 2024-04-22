#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

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
    thrust::device_vector<Bee> d_bees(NUM_BEES);
    thrust::device_vector<int> d_globalBestPosition(NUM_CITIES);
    thrust::device_vector<double> d_globalBestFitness(1);
    thrust::device_vector<curandState> d_state(NUM_BEES);

    Bee* bees = thrust::raw_pointer_cast(d_bees.data());
    int* globalBestPosition = thrust::raw_pointer_cast(d_globalBestPosition.data());
    double* globalBestFitness = thrust::raw_pointer_cast(d_globalBestFitness.data());
    curandState* state = thrust::raw_pointer_cast(d_state.data());

    double initialFitness = INFINITY;
    thrust::copy(&initialFitness, &initialFitness + 1, d_globalBestFitness.begin());

    auto start = std::chrono::high_resolution_clock::now();
    initializeBees<<<(NUM_BEES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(bees, globalBestPosition, globalBestFitness, state);
    cudaDeviceSynchronize();
    runABC(bees, globalBestPosition, globalBestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<int> h_globalBestPosition(NUM_CITIES);
    thrust::host_vector<double> h_globalBestFitness(1);
    thrust::copy(d_globalBestPosition.begin(), d_globalBestPosition.end(), h_globalBestPosition.begin());
    thrust::copy(d_globalBestFitness.begin(), d_globalBestFitness.end(), h_globalBestFitness.begin());

    printResults(h_globalBestPosition.data(), h_globalBestFitness[0], executionTime);

    return 0;
}
