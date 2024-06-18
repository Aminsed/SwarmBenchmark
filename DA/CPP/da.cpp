#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>


struct Dragonfly {
    double position[DIMENSIONS];
    double step[DIMENSIONS];
};

void updateBestFitness(Dragonfly* d, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(d->position);
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = d->position[i];
        }
    }
}

void initializeDragonflies(Dragonfly* dragonflies, double* globalBestPosition, double* globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-5.0, 5.0);
    std::uniform_real_distribution<double> stepDis(-1.0, 1.0);

    for (int i = 0; i < NUM_DRAGONFLIES; i++) {
        Dragonfly* d = &dragonflies[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            d->position[j] = dis(gen);
            d->step[j] = stepDis(gen);
        }
        updateBestFitness(d, globalBestPosition, globalBestFitness);
    }
}

void updateDragonflies(Dragonfly* dragonflies, double* globalBestPosition, double* globalBestFitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < NUM_DRAGONFLIES; i++) {
        Dragonfly* d = &dragonflies[i];
        double separation[DIMENSIONS] = {0};
        double alignment[DIMENSIONS] = {0};
        double cohesion[DIMENSIONS] = {0};
        for (int j = 0; j < NUM_DRAGONFLIES; j++) {
            if (i != j) {
                Dragonfly* neighbor = &dragonflies[j];
                for (int k = 0; k < DIMENSIONS; k++) {
                    double diff = d->position[k] - neighbor->position[k];
                    separation[k] += diff;
                    alignment[k] += neighbor->step[k];
                    cohesion[k] += neighbor->position[k];
                }
            }
        }
        for (int j = 0; j < DIMENSIONS; j++) {
            separation[j] /= (NUM_DRAGONFLIES - 1);
            alignment[j] /= (NUM_DRAGONFLIES - 1);
            cohesion[j] /= (NUM_DRAGONFLIES - 1);

            double r = dis(gen);
            d->step[j] = SEPARATION_WEIGHT * separation[j] +
                         ALIGNMENT_WEIGHT * alignment[j] +
                         COHESION_WEIGHT * (cohesion[j] - d->position[j]) +
                         FOOD_ATTRACTION_WEIGHT * (globalBestPosition[j] - d->position[j]) +
                         ENEMY_DISTRACTION_WEIGHT * (dis(gen) - 0.5);
        }
        for (int j = 0; j < DIMENSIONS; j++) {
            d->position[j] += d->step[j];
        }
        updateBestFitness(d, globalBestPosition, globalBestFitness);
    }
}

void runDragonflyAlgorithm(Dragonfly* dragonflies, double* globalBestPosition, double* globalBestFitness) {
    std::ofstream outputFile("results.txt");
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateDragonflies(dragonflies, globalBestPosition, globalBestFitness);
        outputFile << iter + 1 << ": " << *globalBestFitness << std::endl;
    }
    outputFile.close();
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
    Dragonfly* dragonflies = new Dragonfly[NUM_DRAGONFLIES];
    double* globalBestPosition = new double[DIMENSIONS];
    double globalBestFitness = INFINITY;

    auto start = std::chrono::high_resolution_clock::now();
    initializeDragonflies(dragonflies, globalBestPosition, &globalBestFitness);
    runDragonflyAlgorithm(dragonflies, globalBestPosition, &globalBestFitness);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(globalBestPosition, globalBestFitness, executionTime);

    delete[] dragonflies;
    delete[] globalBestPosition;

    return 0;
}
