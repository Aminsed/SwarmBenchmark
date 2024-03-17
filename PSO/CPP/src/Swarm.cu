#include "Swarm.h"
#include <iostream>
#include <random>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__global__ void updateVelocitiesKernel(Particle* particles, size_t size, const thrust::pair<double, double>& globalBestPosition,
                                       double inertiaWeight, double cognitiveWeight, double socialWeight) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double randP = thrust::uniform_real_distribution<double>(0.0, 1.0)(thrust::default_random_engine());
        double randG = thrust::uniform_real_distribution<double>(0.0, 1.0)(thrust::default_random_engine());
        particles[idx].updateVelocity(globalBestPosition, inertiaWeight, cognitiveWeight, socialWeight, randP, randG);
    }
}

__global__ void updatePositionsKernel(Particle* particles, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        particles[idx].updatePosition();
    }
}

Swarm::Swarm(size_t size, double searchSpaceMin, double searchSpaceMax, std::function<double(double, double)> objectiveFunc)
    : objectiveFunc(objectiveFunc), searchSpaceMin(searchSpaceMin), searchSpaceMax(searchSpaceMax), dis(searchSpaceMin, searchSpaceMax) {
    std::random_device rd;
    gen.seed(rd());
    particles.resize(size);
    thrust::for_each(particles.begin(), particles.end(), [=] __device__ (Particle& particle) {
        particle = Particle(randomDouble(), randomDouble());
    });
    initialize();
}

void Swarm::initialize() {
    thrust::for_each(particles.begin(), particles.end(), [=] __device__ (Particle& particle) {
        particle.evaluateBestPosition(objectiveFunc);
    });
    updateGlobalBest();
}

void Swarm::optimize(int maxIterations) {
    const double inertiaWeight = 0.729, cognitiveWeight = 1.49445, socialWeight = 1.49445;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        size_t blockSize = 256;
        size_t gridSize = (particles.size() + blockSize - 1) / blockSize;
        
        updateVelocitiesKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(particles.data()), particles.size(),
                                                        globalBestPosition, inertiaWeight, cognitiveWeight, socialWeight);
        updatePositionsKernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(particles.data()), particles.size());
        
        thrust::for_each(particles.begin(), particles.end(), [=] __device__ (Particle& particle) {
            particle.evaluateBestPosition(objectiveFunc);
        });
        
        updateGlobalBest();
    }
}

void Swarm::printGlobalBest() const {
    std::cout << "Global Best Position: (" << globalBestPosition.first << ", " << globalBestPosition.second << ")\n";
    std::cout << "Global Best Value: " << globalBestValue << std::endl;
}

void Swarm::updateGlobalBest() {
    thrust::device_ptr<Particle> bestParticle = thrust::min_element(particles.begin(), particles.end(), 
        [] __device__ (const Particle& a, const Particle& b) {
            return a.getBestValue() < b.getBestValue();
        });
    
    if (bestParticle->getBestValue() < globalBestValue) {
        globalBestValue = bestParticle->getBestValue();
        globalBestPosition = bestParticle->getBestPosition();
    }
}

double Swarm::randomDouble() {
    return dis(gen);
}
