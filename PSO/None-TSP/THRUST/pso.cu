// ParticleSwarmOptimization.cu
#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

struct Particle {
    double position[DIMENSIONS];
    double velocity[DIMENSIONS];
    double bestPosition[DIMENSIONS];
    double bestFitness;
};

__device__ void updateBestFitness(Particle* p, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(p->position);

    if (fitness < p->bestFitness) {
        p->bestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            p->bestPosition[i] = p->position[i];
        }
    }

    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = p->position[i];
        }
    }
}

__global__ void initializeParticles(Particle* particles, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_PARTICLES) {
        Particle* p = &particles[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);

        for (int i = 0; i < DIMENSIONS; i++) {
            p->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
            p->velocity[i] = curand_uniform_double(s) * 2.0 - 1.0;
            p->bestPosition[i] = p->position[i];
        }
        p->bestFitness = INFINITY;

        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

__global__ void updateParticles(Particle* particles, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < NUM_PARTICLES) {
        Particle* p = &particles[tid];
        curandState* s = &state[tid];

        for (int i = 0; i < DIMENSIONS; i++) {
            double r1 = curand_uniform_double(s);
            double r2 = curand_uniform_double(s);

            p->velocity[i] = INERTIA_WEIGHT * p->velocity[i] +
                             COGNITIVE_WEIGHT * r1 * (p->bestPosition[i] - p->position[i]) +
                             SOCIAL_WEIGHT * r2 * (globalBestPosition[i] - p->position[i]);

            p->position[i] += p->velocity[i];
        }

        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

void runPSO(Particle* particles, double* globalBestPosition, double* globalBestFitness, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateParticles<<<grid, block>>>(particles, globalBestPosition, globalBestFitness, state);
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
    thrust::device_vector<Particle> d_particles(NUM_PARTICLES);
    thrust::device_vector<double> d_globalBestPosition(DIMENSIONS);
    thrust::device_vector<double> d_globalBestFitness(1);
    thrust::device_vector<curandState> d_state(NUM_PARTICLES);

    double initialFitness = INFINITY;
    d_globalBestFitness[0] = initialFitness;

    auto start = std::chrono::high_resolution_clock::now();

    initializeParticles<<<(NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        thrust::raw_pointer_cast(d_globalBestPosition.data()),
        thrust::raw_pointer_cast(d_globalBestFitness.data()),
        thrust::raw_pointer_cast(d_state.data())
    );
    cudaDeviceSynchronize();

    runPSO(
        thrust::raw_pointer_cast(d_particles.data()),
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
