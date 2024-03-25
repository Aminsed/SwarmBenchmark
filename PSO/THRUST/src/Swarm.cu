#include "Swarm.h"
#include "ObjectiveFunction.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

__global__ void updateParticlesKernel(Particle* particles, size_t size, const double* globalBestPosition,
                                      double inertiaWeight, double cognitiveWeight, double socialWeight,
                                      double* randomP, double* randomG, double (*objectiveFunc)(double, double)) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        particles[index].updateVelocity(globalBestPosition, inertiaWeight, cognitiveWeight, socialWeight, randomP[index], randomG[index]);
        particles[index].updatePosition();
        particles[index].evaluateBestPosition(objectiveFunc);
    }
}

__global__ void initializeParticlesKernel(Particle* particles, size_t size, double* positions) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        particles[index] = Particle(positions[2 * index], positions[2 * index + 1]);
    }
}

__global__ void initializeGlobalBestKernel(const Particle* particles, size_t size, double* globalBestPosition, double* globalBestValue) {
    __shared__ double sharedBestValue;
    __shared__ double sharedBestPosition[2];

    if (threadIdx.x == 0) {
        sharedBestValue = *globalBestValue;
        sharedBestPosition[0] = globalBestPosition[0];
        sharedBestPosition[1] = globalBestPosition[1];
    }
    __syncthreads();

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        double bestValue = particles[index].getBestValue();
        if (bestValue < sharedBestValue) {
            sharedBestValue = bestValue;
            const double* bestPosition = particles[index].getBestPosition();
            sharedBestPosition[0] = bestPosition[0];
            sharedBestPosition[1] = bestPosition[1];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *globalBestValue = sharedBestValue;
        globalBestPosition[0] = sharedBestPosition[0];
        globalBestPosition[1] = sharedBestPosition[1];
    }
}

__global__ void updateGlobalBestKernel(const Particle* particles, size_t size, double* globalBestPosition, double* globalBestValue) {
    __shared__ double sharedBestValue;
    __shared__ double sharedBestPosition[2];

    if (threadIdx.x == 0) {
        sharedBestValue = *globalBestValue;
        sharedBestPosition[0] = globalBestPosition[0];
        sharedBestPosition[1] = globalBestPosition[1];
    }
    __syncthreads();

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        double bestValue = particles[index].getBestValue();
        if (bestValue < sharedBestValue) {
            sharedBestValue = bestValue;
            const double* bestPosition = particles[index].getBestPosition();
            sharedBestPosition[0] = bestPosition[0];
            sharedBestPosition[1] = bestPosition[1];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *globalBestValue = sharedBestValue;
        globalBestPosition[0] = sharedBestPosition[0];
        globalBestPosition[1] = sharedBestPosition[1];
    }
}

Swarm::Swarm(size_t size, double searchSpaceMin, double searchSpaceMax, double (*objectiveFunc)(double, double))
    : objectiveFunc(objectiveFunc), searchSpaceMin(searchSpaceMin), searchSpaceMax(searchSpaceMax), dis(searchSpaceMin, searchSpaceMax) {
    std::random_device rd;
    gen.seed(rd());
    particles.resize(size);
}

void Swarm::initialize() {
    size_t size = particles.size();
    double* positions = new double[2 * size];

    for (size_t i = 0; i < size; ++i) {
        positions[2 * i] = randomDouble();
        positions[2 * i + 1] = randomDouble();
    }

    Particle* d_particles;
    cudaMalloc(&d_particles, size * sizeof(Particle));
    double* d_positions;
    cudaMalloc(&d_positions, 2 * size * sizeof(double));

    cudaMemcpy(d_positions, positions, 2 * size * sizeof(double), cudaMemcpyHostToDevice);

    size_t blockSize = 256;
    size_t gridSize = (size + blockSize - 1) / blockSize;
    initializeParticlesKernel<<<gridSize, blockSize>>>(d_particles, size, d_positions);

    double* d_globalBestPosition;
    cudaMalloc(&d_globalBestPosition, 2 * sizeof(double));
    double* d_globalBestValue;
    cudaMalloc(&d_globalBestValue, sizeof(double));
    cudaMemcpy(d_globalBestValue, &globalBestValue, sizeof(double), cudaMemcpyHostToDevice);

    initializeGlobalBestKernel<<<gridSize, blockSize>>>(d_particles, size, d_globalBestPosition, d_globalBestValue);

    cudaMemcpy(particles.data(), d_particles, size * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(&globalBestValue, d_globalBestValue, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(globalBestPosition, d_globalBestPosition, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    delete[] positions;
    cudaFree(d_particles);
    cudaFree(d_positions);
    cudaFree(d_globalBestPosition);
    cudaFree(d_globalBestValue);
}

void Swarm::optimize(int maxIterations, double* d_globalBestPosition) {
    const double inertiaWeight = 0.729, cognitiveWeight = 1.49445, socialWeight = 1.49445;

    size_t size = particles.size();
    Particle* d_particles;
    cudaMalloc(&d_particles, size * sizeof(Particle));
    cudaMemcpy(d_particles, particles.data(), size * sizeof(Particle), cudaMemcpyHostToDevice);

    double* d_randomP;
    cudaMalloc(&d_randomP, size * sizeof(double));
    double* d_randomG;
    cudaMalloc(&d_randomG, size * sizeof(double));

    std::vector<double> randomP(size);
    std::vector<double> randomG(size);

    for (int iter = 0; iter < maxIterations; ++iter) {
        for (size_t i = 0; i < size; ++i) {
            randomP[i] = randomDouble();
            randomG[i] = randomDouble();
        }

        cudaMemcpy(d_randomP, randomP.data(), size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_randomG, randomG.data(), size * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_globalBestPosition, globalBestPosition, 2 * sizeof(double), cudaMemcpyHostToDevice);

        size_t blockSize = 256;
        size_t gridSize = (size + blockSize - 1) / blockSize;
        updateParticlesKernel<<<gridSize, blockSize>>>(d_particles, size, d_globalBestPosition,
                                                       inertiaWeight, cognitiveWeight, socialWeight,
                                                       d_randomP, d_randomG, objectiveFunc);

        cudaMemcpy(particles.data(), d_particles, size * sizeof(Particle), cudaMemcpyDeviceToHost);

        double* d_globalBestValue;
        cudaMalloc(&d_globalBestValue, sizeof(double));
        cudaMemcpy(d_globalBestValue, &globalBestValue, sizeof(double), cudaMemcpyHostToDevice);

        updateGlobalBestKernel<<<gridSize, blockSize>>>(d_particles, size, d_globalBestPosition, d_globalBestValue);

        cudaMemcpy(&globalBestValue, d_globalBestValue, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(globalBestPosition, d_globalBestPosition, 2 * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_globalBestValue);
    }

    cudaFree(d_particles);
    cudaFree(d_randomP);
    cudaFree(d_randomG);
}

double Swarm::getGlobalBestValue() const {
    return globalBestValue;
}

double Swarm::randomDouble() {
    return dis(gen);
}
