#include "Swarm.h"
#include <iostream>
#include <random>

Swarm::Swarm(size_t size, double searchSpaceMin, double searchSpaceMax, std::function<double(double, double)> objectiveFunc)
    : objectiveFunc(objectiveFunc), searchSpaceMin(searchSpaceMin), searchSpaceMax(searchSpaceMax), dis(searchSpaceMin, searchSpaceMax) {
    std::random_device rd;
    gen.seed(rd());
    for (size_t i = 0; i < size; ++i) {
        particles.emplace_back(Particle(randomDouble(), randomDouble()));
    }
    initialize();
}

void Swarm::initialize() {
    for (auto& particle : particles) {
        particle.evaluateBestPosition(objectiveFunc);
        updateGlobalBest();
    }
}

void Swarm::optimize(int maxIterations) {
    const double inertiaWeight = 0.729, cognitiveWeight = 1.49445, socialWeight = 1.49445;

    for (int iter = 0; iter < maxIterations; ++iter) {
        for (auto& particle : particles) {
            double randP = randomDouble(), randG = randomDouble();
            particle.updateVelocity(globalBestPosition, inertiaWeight, cognitiveWeight, socialWeight, randP, randG);
            particle.updatePosition();
            particle.evaluateBestPosition(objectiveFunc);
            updateGlobalBest();
        }
    }
}

void Swarm::printGlobalBest() const {
    std::cout << "Global Best Position: (" << globalBestPosition.first << ", " << globalBestPosition.second << ")\n";
    std::cout << "Global Best Value: " << globalBestValue << std::endl;
}

void Swarm::updateGlobalBest() {
    for (auto& particle : particles) {
        if (particle.getBestValue() < globalBestValue) {
            globalBestValue = particle.getBestValue();
            globalBestPosition = particle.getBestPosition();
        }
    }
}

double Swarm::randomDouble() {
    return dis(gen);
}
