#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

struct Particle {
    int position[NUM_CITIES];
    double velocity[NUM_CITIES];
    int bestPosition[NUM_CITIES];
    double bestFitness;
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

__device__ void updateBestFitness(Particle* p, int* globalBestPosition, double* globalBestFitness) {
    double fitness = calculateDistance(p->position);
    if (fitness < p->bestFitness) {
        p->bestFitness = fitness;
        for (int i = 0; i < NUM_CITIES; i++) {
            p->bestPosition[i] = p->position[i];
        }
    }
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < NUM_CITIES; i++) {
            globalBestPosition[i] = p->position[i];
        }
    }
}

__global__ void initializeParticles(Particle* particles, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_PARTICLES) {
        Particle* p = &particles[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < NUM_CITIES; i++) {
            p->position[i] = i;
        }
        for (int i = NUM_CITIES - 1; i > 0; i--) {
            int j = curand(s) % (i + 1);
            int temp = p->position[i];
            p->position[i] = p->position[j];
            p->position[j] = temp;
        }
        for (int i = 0; i < NUM_CITIES; i++) {
            p->velocity[i] = curand_uniform_double(s) * 2.0 - 1.0;
            p->bestPosition[i] = p->position[i];
        }
        p->bestFitness = calculateDistance(p->position);
        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

__global__ void updateParticles(Particle* particles, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_PARTICLES) {
        Particle* p = &particles[tid];
        curandState* s = &state[tid];
        for (int i = 0; i < NUM_CITIES; i++) {
            double r1 = curand_uniform_double(s);
            double r2 = curand_uniform_double(s);
            p->velocity[i] = INERTIA_WEIGHT * p->velocity[i] +
                             COGNITIVE_WEIGHT * r1 * (p->bestPosition[i] - p->position[i]) +
                             SOCIAL_WEIGHT * r2 * (globalBestPosition[i] - p->position[i]);
        }
        int newPosition[NUM_CITIES];
        for (int i = 0; i < NUM_CITIES; i++) {
            newPosition[i] = p->position[i];
        }
        for (int i = 0; i < NUM_CITIES; i++) {
            int swapIndex = (static_cast<int>(p->velocity[i]) + NUM_CITIES) % NUM_CITIES;
            int temp = newPosition[i];
            newPosition[i] = newPosition[swapIndex];
            newPosition[swapIndex] = temp;
        }
        bool isValid = true;
        for (int i = 0; i < NUM_CITIES; i++) {
            bool found = false;
            for (int j = 0; j < NUM_CITIES; j++) {
                if (newPosition[j] == i) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                isValid = false;
                break;
            }
        }
        if (isValid) {
            for (int i = 0; i < NUM_CITIES; i++) {
                p->position[i] = newPosition[i];
            }
            updateBestFitness(p, globalBestPosition, globalBestFitness);
        }
    }
}

void runPSO(Particle* particles, int* globalBestPosition, double* globalBestFitness, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateParticles<<<grid, block>>>(particles, globalBestPosition, globalBestFitness, state);
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
    thrust::device_vector<Particle> d_particles(NUM_PARTICLES);
    thrust::device_vector<int> d_globalBestPosition(NUM_CITIES);
    thrust::device_vector<double> d_globalBestFitness(1);
    thrust::device_vector<curandState> d_state(NUM_PARTICLES);

    Particle* particles = thrust::raw_pointer_cast(d_particles.data());
    int* globalBestPosition = thrust::raw_pointer_cast(d_globalBestPosition.data());
    double* globalBestFitness = thrust::raw_pointer_cast(d_globalBestFitness.data());
    curandState* state = thrust::raw_pointer_cast(d_state.data());

    double initialFitness = INFINITY;
    thrust::copy(&initialFitness, &initialFitness + 1, d_globalBestFitness.begin());

    auto start = std::chrono::high_resolution_clock::now();
    initializeParticles<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(particles, globalBestPosition, globalBestFitness, state);
    cudaDeviceSynchronize();
    runPSO(particles, globalBestPosition, globalBestFitness, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<int> h_globalBestPosition = d_globalBestPosition;
    thrust::host_vector<double> h_globalBestFitness = d_globalBestFitness;

    printResults(h_globalBestPosition.data(), h_globalBestFitness[0], executionTime);

    return 0;
}
