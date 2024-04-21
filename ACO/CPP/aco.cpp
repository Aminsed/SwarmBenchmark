#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

struct Ant {
    double position[DIMENSIONS];
    double fitness;
};

void updatePheromone(double* pheromone, double* bestPosition, double bestFitness) {
    for (int i = 0; i < DIMENSIONS; i++) {
        pheromone[i] += Q / bestFitness;
    }
}

void initializeAnts(Ant* ants, double* pheromone, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < NUM_ANTS; i++) {
        Ant& a = ants[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            a.position[j] = dist(rng);
        }
        a.fitness = objectiveFunction(a.position);
    }
}

void updateAnts(Ant* ants, double* pheromone, double* bestPosition, double& bestFitness, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> stepDist(-1.0, 1.0);
    for (int i = 0; i < NUM_ANTS; i++) {
        Ant& a = ants[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            double r = dist(rng);
            if (r < PHEROMONE_WEIGHT) {
                a.position[j] = bestPosition[j];
            } else {
                a.position[j] += stepDist(rng);
            }
        }
        a.fitness = objectiveFunction(a.position);
        if (a.fitness < bestFitness) {
            bestFitness = a.fitness;
            for (int j = 0; j < DIMENSIONS; j++) {
                bestPosition[j] = a.position[j];
            }
            updatePheromone(pheromone, bestPosition, bestFitness);
        }
    }
}

void runACO(Ant* ants, double* pheromone, double* bestPosition, double& bestFitness, std::mt19937& rng) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateAnts(ants, pheromone, bestPosition, bestFitness, rng);
    }
}

void printResults(double* bestPosition, double bestFitness, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Best Position: " << bestPosition[0] << std::endl;
    } else {
        std::cout << "Best Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << bestPosition[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Value: " << bestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    Ant* ants = new Ant[NUM_ANTS];
    double* pheromone = new double[DIMENSIONS];
    double* bestPosition = new double[DIMENSIONS];
    double bestFitness = INFINITY;
    std::random_device rd;
    std::mt19937 rng(rd());

    auto start = std::chrono::high_resolution_clock::now();
    initializeAnts(ants, pheromone, rng);
    runACO(ants, pheromone, bestPosition, bestFitness, rng);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(bestPosition, bestFitness, executionTime);

    delete[] ants;
    delete[] pheromone;
    delete[] bestPosition;

    return 0;
}
