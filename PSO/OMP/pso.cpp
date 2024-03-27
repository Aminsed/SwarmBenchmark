#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <omp.h>

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> bestPosition;
    double bestFitness;
};

void updateBestFitness(Particle& p, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    double fitness = objectiveFunction(p.position);

    if (fitness < p.bestFitness) {
        p.bestFitness = fitness;
        p.bestPosition = p.position;
    }

    #pragma omp critical
    {
        if (fitness < globalBestFitness) {
            globalBestFitness = fitness;
            globalBestPosition = p.position;
        }
    }
}

void initializeParticles(std::vector<Particle>& particles, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        auto& p = particles[i];
        p.position.resize(DIMENSIONS);
        p.velocity.resize(DIMENSIONS);
        p.bestPosition.resize(DIMENSIONS);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-5.0, 5.0);
        std::uniform_real_distribution<> disVel(-1.0, 1.0);

        for (int j = 0; j < DIMENSIONS; ++j) {
            p.position[j] = dis(gen);
            p.velocity[j] = disVel(gen);
            p.bestPosition[j] = p.position[j];
        }
        p.bestFitness = INFINITY;

        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

void updateParticles(std::vector<Particle>& particles, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        auto& p = particles[i];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int j = 0; j < DIMENSIONS; ++j) {
            double r1 = dis(gen);
            double r2 = dis(gen);

            p.velocity[j] = INERTIA_WEIGHT * p.velocity[j] +
                            COGNITIVE_WEIGHT * r1 * (p.bestPosition[j] - p.position[j]) +
                            SOCIAL_WEIGHT * r2 * (globalBestPosition[j] - p.position[j]);

            p.position[j] += p.velocity[j];
        }

        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

void runPSO(std::vector<Particle>& particles, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        updateParticles(particles, globalBestPosition, globalBestFitness);
    }
}

void printResults(const std::vector<double>& globalBestPosition, double globalBestFitness, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Global Best Position: " << globalBestPosition[0] << std::endl;
    } else {
        std::cout << "Global Best Position: (";
        for (int i = 0; i < DIMENSIONS; ++i) {
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
    std::vector<Particle> particles(NUM_PARTICLES);
    std::vector<double> globalBestPosition(DIMENSIONS);
    double globalBestFitness = INFINITY;

    auto start = std::chrono::high_resolution_clock::now();

    initializeParticles(particles, globalBestPosition, globalBestFitness);
    runPSO(particles, globalBestPosition, globalBestFitness);

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(globalBestPosition, globalBestFitness, executionTime);

    return 0;
}
