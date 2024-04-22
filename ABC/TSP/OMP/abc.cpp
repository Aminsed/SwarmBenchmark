#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

struct Bee {
    int position[NUM_CITIES];
    double fitness;
};

double calculateDistance(int* position) {
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

void updateBestFitness(Bee* b, int* globalBestPosition, double* globalBestFitness) {
    double fitness = calculateDistance(b->position);
    b->fitness = fitness;
    #pragma omp critical
    {
        if (fitness < *globalBestFitness) {
            *globalBestFitness = fitness;
            for (int i = 0; i < NUM_CITIES; i++) {
                globalBestPosition[i] = b->position[i];
            }
        }
    }
}

void initializeBees(Bee* bees, int* globalBestPosition, double* globalBestFitness) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_BEES; tid++) {
        Bee* b = &bees[tid];
        unsigned int seed = omp_get_thread_num();
        std::mt19937 rng(seed);
        for (int i = 0; i < NUM_CITIES; i++) {
            b->position[i] = i;
        }
        std::shuffle(b->position, b->position + NUM_CITIES, rng);
        updateBestFitness(b, globalBestPosition, globalBestFitness);
    }
}

void sendEmployedBees(Bee* bees, int* globalBestPosition, double* globalBestFitness) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_BEES; tid++) {
        Bee* b = &bees[tid];
        unsigned int seed = omp_get_thread_num();
        std::mt19937 rng(seed);
        int newPosition[NUM_CITIES];
        std::copy(b->position, b->position + NUM_CITIES, newPosition);
        std::uniform_int_distribution<int> dist(0, NUM_CITIES - 1);
        int i = dist(rng);
        int j = dist(rng);
        std::swap(newPosition[i], newPosition[j]);
        double newFitness = calculateDistance(newPosition);
        if (newFitness < b->fitness) {
            std::copy(newPosition, newPosition + NUM_CITIES, b->position);
            updateBestFitness(b, globalBestPosition, globalBestFitness);
        }
    }
}

void sendOnlookerBees(Bee* bees, int* globalBestPosition, double* globalBestFitness) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_BEES; tid++) {
        unsigned int seed = omp_get_thread_num();
        std::mt19937 rng(seed);
        double probabilities[NUM_BEES];
        double maxFitness = bees[0].fitness;
        for (int i = 1; i < NUM_BEES; i++) {
            if (bees[i].fitness > maxFitness) {
                maxFitness = bees[i].fitness;
            }
        }
        double fitnessSum = 0.0;
        for (int i = 0; i < NUM_BEES; i++) {
            probabilities[i] = (0.9 * (maxFitness - bees[i].fitness) / maxFitness) + 0.1;
            fitnessSum += probabilities[i];
        }
        std::uniform_real_distribution<double> dist(0.0, fitnessSum);
        double r = dist(rng);
        double cumulativeProbability = 0.0;
        int selectedBee = -1;
        for (int i = 0; i < NUM_BEES; i++) {
            cumulativeProbability += probabilities[i];
            if (r <= cumulativeProbability) {
                selectedBee = i;
                break;
            }
        }
        if (selectedBee != -1) {
            int newPosition[NUM_CITIES];
            std::copy(bees[selectedBee].position, bees[selectedBee].position + NUM_CITIES, newPosition);
            std::uniform_int_distribution<int> dist(0, NUM_CITIES - 1);
            int i = dist(rng);
            int j = dist(rng);
            std::swap(newPosition[i], newPosition[j]);
            double newFitness = calculateDistance(newPosition);
            if (newFitness < bees[selectedBee].fitness) {
                std::copy(newPosition, newPosition + NUM_CITIES, bees[selectedBee].position);
                updateBestFitness(&bees[selectedBee], globalBestPosition, globalBestFitness);
            }
        }
    }
}

void sendScoutBees(Bee* bees, int* globalBestPosition, double* globalBestFitness) {
    #pragma omp parallel for
    for (int tid = 0; tid < NUM_BEES; tid++) {
        Bee* b = &bees[tid];
        unsigned int seed = omp_get_thread_num();
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < SCOUT_BEE_PROBABILITY) {
            for (int i = 0; i < NUM_CITIES; i++) {
                b->position[i] = i;
            }
            std::shuffle(b->position, b->position + NUM_CITIES, rng);
            updateBestFitness(b, globalBestPosition, globalBestFitness);
        }
    }
}

void runABC(Bee* bees, int* globalBestPosition, double* globalBestFitness) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        sendEmployedBees(bees, globalBestPosition, globalBestFitness);
        sendOnlookerBees(bees, globalBestPosition, globalBestFitness);
        sendScoutBees(bees, globalBestPosition, globalBestFitness);
    }
}

void printResults(int* globalBestPosition, double globalBestFitness, double executionTime) {
    std::cout << "Best Path: ";
    for (int i = 0; i < NUM_CITIES; i++) {
        std::cout << globalBestPosition[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Best Distance: " << globalBestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    Bee* bees = new Bee[NUM_BEES];
    int* globalBestPosition = new int[NUM_CITIES];
    double globalBestFitness = INFINITY;

    auto start = std::chrono::high_resolution_clock::now();
    initializeBees(bees, globalBestPosition, &globalBestFitness);
    runABC(bees, globalBestPosition, &globalBestFitness);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(globalBestPosition, globalBestFitness, executionTime);

    delete[] bees;
    delete[] globalBestPosition;

    return 0;
}
