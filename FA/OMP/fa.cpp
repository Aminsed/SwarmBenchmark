#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

struct Firefly {
    std::vector<double> position;
    double brightness;
};

double attractiveness(double distance) {
    return BETA0 * exp(-GAMMA * distance * distance);
}

void initializeFireflies(std::vector<Firefly>& fireflies, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-5.0, 5.0);

    #pragma omp parallel for
    for (int i = 0; i < NUM_FIREFLIES; i++) {
        auto& firefly = fireflies[i];
        firefly.position.resize(DIMENSIONS);
        for (int j = 0; j < DIMENSIONS; j++) {
            firefly.position[j] = dis(gen);
        }
        firefly.brightness = objectiveFunction(firefly.position);

        #pragma omp critical
        {
            if (firefly.brightness < globalBestFitness) {
                globalBestFitness = firefly.brightness;
                globalBestPosition = firefly.position;
            }
        }
    }
}

void updateFireflies(std::vector<Firefly>& fireflies, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    #pragma omp parallel for
    for (int i = 0; i < NUM_FIREFLIES; i++) {
        auto& firefly = fireflies[i];
        for (int j = 0; j < NUM_FIREFLIES; j++) {
            if (i != j) {
                const auto& otherFirefly = fireflies[j];
                double distance = 0.0;
                for (int k = 0; k < DIMENSIONS; k++) {
                    double diff = firefly.position[k] - otherFirefly.position[k];
                    distance += diff * diff;
                }
                distance = sqrt(distance);
                double beta = attractiveness(distance);
                if (otherFirefly.brightness > firefly.brightness) {
                    for (int k = 0; k < DIMENSIONS; k++) {
                        double r = dis(gen);
                        firefly.position[k] += beta * (otherFirefly.position[k] - firefly.position[k]) + ALPHA * (r - 0.5);
                    }
                }
            }
        }
        firefly.brightness = objectiveFunction(firefly.position);

        #pragma omp critical
        {
            if (firefly.brightness < globalBestFitness) {
                globalBestFitness = firefly.brightness;
                globalBestPosition = firefly.position;
            }
        }
    }
}

void runFA(std::vector<Firefly>& fireflies, std::vector<double>& globalBestPosition, double& globalBestFitness) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateFireflies(fireflies, globalBestPosition, globalBestFitness);
    }
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
    std::vector<Firefly> fireflies(NUM_FIREFLIES);
    std::vector<double> globalBestPosition(DIMENSIONS);
    double globalBestFitness = std::numeric_limits<double>::infinity();

    auto start = std::chrono::high_resolution_clock::now();
    initializeFireflies(fireflies, globalBestPosition, globalBestFitness);
    runFA(fireflies, globalBestPosition, globalBestFitness);
    auto end = std::chrono::high_resolution_clock::now();

    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printResults(globalBestPosition, globalBestFitness, executionTime);

    return 0;
}
