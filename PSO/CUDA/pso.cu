#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int NUM_PARTICLES = 1000;
constexpr int MAX_ITERATIONS = 100;
constexpr double SEARCH_SPACE_MIN = -5.12;
constexpr double SEARCH_SPACE_MAX = 5.12;
constexpr double INERTIA_WEIGHT = 0.729;
constexpr double COGNITIVE_WEIGHT = 1.49;
constexpr double SOCIAL_WEIGHT = 1.49;

struct Particle {
    double position[2];
    double velocity[2];
    double bestPosition[2];
    double bestValue;
};

__device__ double objectiveFunction(const double* position) {
    double x = position[0];
    double y = position[1];
    return 20 + x * x - 10 * cos(2 * M_PI * x) + y * y - 10 * cos(2 * M_PI * y);
}

__global__ void initializeRandStates(curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        curand_init(clock64(), idx, 0, &randStates[idx]);
    }
}

__global__ void updateParticles(Particle* particles, double* globalBestPosition, double* globalBestValue, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        Particle& particle = particles[idx];

        particle.bestValue = min(particle.bestValue, objectiveFunction(particle.position));
        if (particle.bestValue < objectiveFunction(particle.bestPosition)) {
            particle.bestPosition[0] = particle.position[0];
            particle.bestPosition[1] = particle.position[1];
        }

        if (particle.bestValue < *globalBestValue) {
            globalBestPosition[0] = particle.bestPosition[0];
            globalBestPosition[1] = particle.bestPosition[1];
            *globalBestValue = particle.bestValue;
        }

        double r1 = curand_uniform(&randStates[idx]);
        double r2 = curand_uniform(&randStates[idx]);

        particle.velocity[0] = INERTIA_WEIGHT * particle.velocity[0] +
                               COGNITIVE_WEIGHT * r1 * (particle.bestPosition[0] - particle.position[0]) +
                               SOCIAL_WEIGHT * r2 * (globalBestPosition[0] - particle.position[0]);
        particle.velocity[1] = INERTIA_WEIGHT * particle.velocity[1] +
                               COGNITIVE_WEIGHT * r1 * (particle.bestPosition[1] - particle.position[1]) +
                               SOCIAL_WEIGHT * r2 * (globalBestPosition[1] - particle.position[1]);

        particle.position[0] += particle.velocity[0];
        particle.position[1] += particle.velocity[1];
    }
}

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(SEARCH_SPACE_MIN, SEARCH_SPACE_MAX);

    Particle* particles;
    cudaMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle));

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        particles[i].position[0] = dist(gen);
        particles[i].position[1] = dist(gen);
        particles[i].velocity[0] = dist(gen);
        particles[i].velocity[1] = dist(gen);
        particles[i].bestPosition[0] = 0;
        particles[i].bestPosition[1] = 0;
        particles[i].bestValue = std::numeric_limits<double>::max();
    }

    double* globalBestPosition;
    cudaMallocManaged(&globalBestPosition, 2 * sizeof(double));
    globalBestPosition[0] = 0;
    globalBestPosition[1] = 0;

    double* globalBestValue;
    cudaMallocManaged(&globalBestValue, sizeof(double));
    *globalBestValue = std::numeric_limits<double>::max();

    curandState* randStates;
    cudaMalloc(&randStates, NUM_PARTICLES * sizeof(curandState));

    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

    initializeRandStates<<<numBlocks, blockSize>>>(randStates);
    cudaDeviceSynchronize();

    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        updateParticles<<<numBlocks, blockSize>>>(particles, globalBestPosition, globalBestValue, randStates);
        cudaDeviceSynchronize();
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Global Best Position: (" << globalBestPosition[0] << ", " << globalBestPosition[1] << ")\n";
    std::cout << "Global Best Value: " << *globalBestValue << "\n";
    std::cout << "Execution Time: " << duration.count() << " milliseconds\n";

    cudaFree(particles);
    cudaFree(globalBestPosition);
    cudaFree(globalBestValue);
    cudaFree(randStates);

    return 0;
}
