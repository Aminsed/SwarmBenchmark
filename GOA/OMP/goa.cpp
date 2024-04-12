#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <omp.h>

struct Grasshopper {
    double position[DIMENSIONS];
};

void updateBestFitness(Grasshopper& g, double* globalBestPosition, double& globalBestFitness) {
    double fitness = objectiveFunction(g.position);
    if (fitness < globalBestFitness) {
        globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = g.position[i];
        }
    }
}

void initializeGrasshoppers(std::vector<Grasshopper>& grasshoppers, double* globalBestPosition, double& globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-5.0, 5.0);

    #pragma omp parallel for
    for (int i = 0; i < NUM_GRASSHOPPERS; i++) {
        Grasshopper& g = grasshoppers[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            g.position[j] = dis(gen);
        }
        updateBestFitness(g, globalBestPosition, globalBestFitness);
    }
}

void updateGrasshoppers(std::vector<Grasshopper>& grasshoppers, double* globalBestPosition, double& globalBestFitness, int iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double c = 0.00001;
    double cMax = 1.0;
    double cMin = 0.00001;
    double l = (cMax - cMin) * (MAX_ITERATIONS - iter) / MAX_ITERATIONS + cMin;

    #pragma omp parallel for
    for (int i = 0; i < NUM_GRASSHOPPERS; i++) {
        Grasshopper& g = grasshoppers[i];
        for (int d = 0; d < DIMENSIONS; d++) {
            double socialInteraction = 0.0;
            for (int j = 0; j < NUM_GRASSHOPPERS; j++) {
                if (j != i) {
                    double distance = fabs(g.position[d] - grasshoppers[j].position[d]);
                    double r = dis(gen);
                    double si = (0.5 + 0.5 * r) * (globalBestPosition[d] - l * distance);
                    socialInteraction += si;
                }
            }
            double xi = c * socialInteraction;
            double r = dis(gen);
            double levy = pow(r, -1.0 / LEVY_EXPONENT);
            double newPosition = g.position[d] + xi * levy;
            g.position[d] = fmax(LOWER_BOUND, fmin(newPosition, UPPER_BOUND));
        }
        updateBestFitness(g, globalBestPosition, globalBestFitness);
    }
}

void runGOA(std::vector<Grasshopper>& grasshoppers, double* globalBestPosition, double& globalBestFitness) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateGrasshoppers(grasshoppers, globalBestPosition, globalBestFitness, iter);
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
    std::vector<Grasshopper> grasshoppers(NUM_GRASSHOPPERS);
    double globalBestPosition[DIMENSIONS];
    double globalBestFitness = INFINITY;

    auto start = std::chrono::high_resolution_clock::now();
    initializeGrasshoppers(grasshoppers, globalBestPosition, globalBestFitness);
    runGOA(grasshoppers, globalBestPosition, globalBestFitness);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(globalBestPosition, globalBestFitness, executionTime);

    return 0;
}
