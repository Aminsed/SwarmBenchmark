#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <random>
#include <fstream>


struct Salp {
    double position[DIMENSIONS];
    double bestPosition[DIMENSIONS];
    double bestFitness;
};

void updateBestFitness(Salp* s, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(s->position);
    if (fitness < s->bestFitness) {
        s->bestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            s->bestPosition[i] = s->position[i];
        }
    }
    #pragma omp critical
    {
        if (fitness < *globalBestFitness) {
            *globalBestFitness = fitness;
            for (int i = 0; i < DIMENSIONS; i++) {
                globalBestPosition[i] = s->position[i];
            }
        }
    }
}

void initializeSalps(Salp* salps, double* globalBestPosition, double* globalBestFitness) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_SALPS; tid++) {
        Salp* s = &salps[tid];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < DIMENSIONS; i++) {
            s->position[i] = dis(gen) * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND;
            s->bestPosition[i] = s->position[i];
        }
        s->bestFitness = INFINITY;
        updateBestFitness(s, globalBestPosition, globalBestFitness);
    }
}

void updateSalps(Salp* salps, double* globalBestPosition, double* globalBestFitness, int iter) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_SALPS; tid++) {
        Salp* s = &salps[tid];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double c1 = 2.0 * exp(-(4.0 * iter / MAX_ITERATIONS) * (4.0 * iter / MAX_ITERATIONS));
        double c2 = dis(gen);
        double c3 = dis(gen);
        for (int i = 0; i < DIMENSIONS; i++) {
            if (tid == 0) {
                s->position[i] = globalBestPosition[i] + c1 * ((UPPER_BOUND - LOWER_BOUND) * c2 + LOWER_BOUND);
            } else {
                s->position[i] = 0.5 * (s->position[i] + globalBestPosition[i]) + c1 * ((UPPER_BOUND - LOWER_BOUND) * c3 + LOWER_BOUND);
            }
        }
        updateBestFitness(s, globalBestPosition, globalBestFitness);
    }
}

void runSSA(Salp* salps, double* globalBestPosition, double* globalBestFitness) {
    std::ofstream outputFile("results.txt");
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateSalps(salps, globalBestPosition, globalBestFitness, iter);
        #pragma omp critical
        {
            outputFile << iter + 1 << ": " << *globalBestFitness << std::endl;
        }
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
    Salp* salps = new Salp[NUM_SALPS];
    double* globalBestPosition = new double[DIMENSIONS];
    double globalBestFitness = INFINITY;

    auto start = std::chrono::high_resolution_clock::now();
    initializeSalps(salps, globalBestPosition, &globalBestFitness);
    runSSA(salps, globalBestPosition, &globalBestFitness);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(globalBestPosition, globalBestFitness, executionTime);

    delete[] salps;
    delete[] globalBestPosition;

    return 0;
}
