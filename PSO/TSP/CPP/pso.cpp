#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>

struct Particle {
    std::vector<int> position;
    std::vector<double> velocity;
    std::vector<int> bestPosition;
    double bestFitness;
};

double calculateDistance(const std::vector<int>& position) {
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

void updateBestFitness(Particle& p, std::vector<int>& globalBestPosition, double& globalBestFitness) {
    double fitness = calculateDistance(p.position);
    if (fitness < p.bestFitness) {
        p.bestFitness = fitness;
        p.bestPosition = p.position;
    }
    if (fitness < globalBestFitness) {
        globalBestFitness = fitness;
        globalBestPosition = p.position;
    }
}

void initializeParticles(std::vector<Particle>& particles, std::vector<int>& globalBestPosition, double& globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& p : particles) {
        p.position.resize(NUM_CITIES);
        std::iota(p.position.begin(), p.position.end(), 0);
        std::shuffle(p.position.begin(), p.position.end(), gen);

        p.velocity.resize(NUM_CITIES);
        for (auto& v : p.velocity) {
            v = dis(gen) * 2.0 - 1.0;
        }

        p.bestPosition = p.position;
        p.bestFitness = calculateDistance(p.position);
        updateBestFitness(p, globalBestPosition, globalBestFitness);
    }
}

void updateParticles(std::vector<Particle>& particles, std::vector<int>& globalBestPosition, double& globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& p : particles) {
        for (int i = 0; i < NUM_CITIES; i++) {
            double r1 = dis(gen);
            double r2 = dis(gen);
            p.velocity[i] = INERTIA_WEIGHT * p.velocity[i] +
                            COGNITIVE_WEIGHT * r1 * (p.bestPosition[i] - p.position[i]) +
                            SOCIAL_WEIGHT * r2 * (globalBestPosition[i] - p.position[i]);
        }

        std::vector<int> newPosition = p.position;
        for (int i = 0; i < NUM_CITIES; i++) {
            int swapIndex = (static_cast<int>(std::abs(p.velocity[i])) + i) % NUM_CITIES;
            std::swap(newPosition[i], newPosition[swapIndex]);
        }

        bool isValid = true;
        for (int i = 0; i < NUM_CITIES; i++) {
            if (std::find(newPosition.begin(), newPosition.end(), i) == newPosition.end()) {
                isValid = false;
                break;
            }
        }

        if (isValid) {
            p.position = newPosition;
            updateBestFitness(p, globalBestPosition, globalBestFitness);
        }
    }
}


void runPSO(std::vector<Particle>& particles, std::vector<int>& globalBestPosition, double& globalBestFitness) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateParticles(particles, globalBestPosition, globalBestFitness);
    }
}

void printResults(const std::vector<int>& globalBestPosition, double globalBestFitness, double executionTime) {
    std::cout << "Best Path: ";
    for (int city : globalBestPosition) {
        std::cout << city << " ";
    }
    std::cout << std::endl;
    std::cout << "Best Distance: " << globalBestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    std::vector<Particle> particles(NUM_PARTICLES);
    std::vector<int> globalBestPosition(NUM_CITIES);
    double globalBestFitness = INFINITY;

    auto start = std::chrono::high_resolution_clock::now();
    initializeParticles(particles, globalBestPosition, globalBestFitness);
    runPSO(particles, globalBestPosition, globalBestFitness);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(globalBestPosition, globalBestFitness, executionTime);

    return 0;
}
