// ParticleSwarmOptimization.cpp
#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>

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

    if (fitness < globalBestFitness) {
        globalBestFitness = fitness;
        globalBestPosition = p.position;
    }
}

void initializeParticles(std::vector<Particle>& particles, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-5.0, 5.0);
    std::uniform_real_distribution<> disVel(-1.0, 1.0);

    for (auto& p : particles) {
        p.position.resize(DIMENSIONS);
        p.velocity.resize(DIMENSIONS);
        p.bestPosition.resize(DIMENSIONS);

        for (int i = 0; i < DIMENSIONS; i++) {
            p.position[i] = dis(gen);
            p.velocity[i] = disVel(gen);
            p.bestPosition[i] = p.position[i];
        }
        p.bestFitness = INFINITY;

        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

void updateParticles(std::vector<Particle>& particles, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& p : particles) {
        for (int i = 0; i < DIMENSIONS; i++) {
            double r1 = dis(gen);
            double r2 = dis(gen);

            p.velocity[i] = INERTIA_WEIGHT * p.velocity[i] +
                            COGNITIVE_WEIGHT * r1 * (p.bestPosition[i] - p.position[i]) +
                            SOCIAL_WEIGHT * r2 * (globalBestPosition[i] - p.position[i]);

            p.position[i] += p.velocity[i];
        }

        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

void runPSO(std::vector<Particle>& particles, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    std::ofstream outputFile("results.txt");

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateParticles(particles, globalBestPosition, globalBestFitness);

        // Write the global best fitness value of each iteration to the file
        outputFile << iter + 1 << ": " << globalBestFitness << std::endl;
    }

    outputFile.close();
}

void printResults(const std::vector<double>& globalBestPosition, double globalBestFitness, double executionTime) {
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
