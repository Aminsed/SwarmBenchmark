#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <omp.h>
#include <fstream>


struct Ant {
    double position[DIMENSIONS];
    double fitness;
};

void updatePheromone(double* pheromone, double* bestPosition, double bestFitness) {
    for (int i = 0; i < DIMENSIONS; i++) {
        pheromone[i] += Q / (bestFitness + 1e-10);
    }
}

void initializeAnts(Ant* ants, double* pheromone, std::mt19937* rng) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_ANTS; tid++) {
        Ant* a = &ants[tid];
        std::uniform_real_distribution<double> dist(-5.0, 5.0);
        for (int i = 0; i < DIMENSIONS; i++) {
            a->position[i] = dist(rng[omp_get_thread_num()]);
            pheromone[i] = 1.0;
        }
        a->fitness = objectiveFunction(a->position);
    }
}


void updateAnts(Ant* ants, double* pheromone, double* bestPosition, double* bestFitness, std::mt19937* rng) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_ANTS; tid++) {
        Ant* a = &ants[tid];
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < DIMENSIONS; i++) {
            double r = dist(rng[omp_get_thread_num()]);
            if (r < PHEROMONE_WEIGHT) {
                a->position[i] = bestPosition[i] + (dist(rng[omp_get_thread_num()]) * 2.0 - 1.0);
            } else {
                a->position[i] += dist(rng[omp_get_thread_num()]) * 2.0 - 1.0;
            }
        }
        a->fitness = objectiveFunction(a->position);
        #pragma omp critical
        {
            if (a->fitness < *bestFitness) {
                *bestFitness = a->fitness;
                for (int i = 0; i < DIMENSIONS; i++) {
                    bestPosition[i] = a->position[i];
                }
            }
        }
    }
    updatePheromone(pheromone, bestPosition, *bestFitness);
}

void runACO(Ant* ants, double* pheromone, double* bestPosition, double* bestFitness, std::mt19937* rng) {
    std::ofstream outputFile("results.txt");
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateAnts(ants, pheromone, bestPosition, bestFitness, rng);
        outputFile << iter + 1 << ": " << *bestFitness << std::endl;
    }
    outputFile.close();
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
    std::mt19937* rng = new std::mt19937[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); i++) {
        rng[i].seed(std::random_device{}());
    }
    auto start = std::chrono::high_resolution_clock::now();
    initializeAnts(ants, pheromone, rng);
    runACO(ants, pheromone, bestPosition, &bestFitness, rng);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printResults(bestPosition, bestFitness, executionTime);
    delete[] ants;
    delete[] pheromone;
    delete[] bestPosition;
    delete[] rng;
    return 0;
}
