#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

struct Ant {
    int tour[NUM_CITIES];
    double tourLength;
};

double calculateTourLength(int* tour) {
    double length = 0.0;
    for (int i = 0; i < NUM_CITIES - 1; i++) {
        int city1 = tour[i];
        int city2 = tour[i + 1];
        length += distances[city1][city2];
    }
    int lastCity = tour[NUM_CITIES - 1];
    int firstCity = tour[0];
    length += distances[lastCity][firstCity];
    return length;
}

void constructTours(Ant* ants, double* pheromones, int* bestTour, double* bestTourLength) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 rng(rd());

        #pragma omp for
        for (int tid = 0; tid < NUM_ANTS; ++tid) {
            Ant* ant = &ants[tid];
            int visited[NUM_CITIES] = {0};
            int current = rng() % NUM_CITIES;
            ant->tour[0] = current;
            visited[current] = 1;
            for (int i = 1; i < NUM_CITIES; i++) {
                double probabilities[NUM_CITIES] = {0};
                double sum = 0.0;
                for (int j = 0; j < NUM_CITIES; j++) {
                    if (!visited[j]) {
                        probabilities[j] = pow(pheromones[current * NUM_CITIES + j], ALPHA) *
                                           pow(1.0 / distances[current][j], BETA);
                        sum += probabilities[j];
                    }
                }
                std::uniform_real_distribution<double> dist(0.0, sum);
                double r = dist(rng);
                double cumulativeProb = 0.0;
                int nextCity = -1;
                for (int j = 0; j < NUM_CITIES; j++) {
                    if (!visited[j]) {
                        cumulativeProb += probabilities[j];
                        if (r <= cumulativeProb) {
                            nextCity = j;
                            break;
                        }
                    }
                }
                ant->tour[i] = nextCity;
                visited[nextCity] = 1;
                current = nextCity;
            }
            ant->tourLength = calculateTourLength(ant->tour);

            #pragma omp critical
            {
                if (ant->tourLength < *bestTourLength) {
                    *bestTourLength = ant->tourLength;
                    for (int i = 0; i < NUM_CITIES; i++) {
                        bestTour[i] = ant->tour[i];
                    }
                }
            }
        }
    }
}

void updatePheromones(double* pheromones, Ant* ants) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            pheromones[i * NUM_CITIES + j] *= EVAPORATION_RATE;
            for (int k = 0; k < NUM_ANTS; k++) {
                Ant* ant = &ants[k];
                for (int l = 0; l < NUM_CITIES - 1; l++) {
                    if (ant->tour[l] == i && ant->tour[l + 1] == j) {
                        pheromones[i * NUM_CITIES + j] += Q / ant->tourLength;
                    }
                }
                if (ant->tour[NUM_CITIES - 1] == i && ant->tour[0] == j) {
                    pheromones[i * NUM_CITIES + j] += Q / ant->tourLength;
                }
            }
        }
    }
}

void runACO(Ant* ants, double* pheromones, int* bestTour, double* bestTourLength) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        constructTours(ants, pheromones, bestTour, bestTourLength);
        updatePheromones(pheromones, ants);
    }
}

void printResults(int* bestTour, double bestTourLength, double executionTime) {
    std::cout << "Best Tour: ";
    for (int i = 0; i < NUM_CITIES; i++) {
        std::cout << bestTour[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Best Tour Length: " << bestTourLength << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    Ant* ants = new Ant[NUM_ANTS];
    double* pheromones = new double[NUM_CITIES * NUM_CITIES];
    int* bestTour = new int[NUM_CITIES];
    double bestTourLength = INFINITY;

    std::fill(pheromones, pheromones + NUM_CITIES * NUM_CITIES, 1.0);

    auto start = std::chrono::high_resolution_clock::now();
    runACO(ants, pheromones, bestTour, &bestTourLength);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(bestTour, bestTourLength, executionTime);

    delete[] ants;
    delete[] pheromones;
    delete[] bestTour;

    return 0;
}
