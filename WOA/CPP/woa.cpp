#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <fstream>

struct Whale {
    double position[DIMENSIONS];
    double bestPosition[DIMENSIONS];
    double bestFitness;
};

void updateBestFitness(Whale* w, double* globalBestPosition, double* globalBestFitness) {
    double fitness = objectiveFunction(w->position);
    if (fitness < w->bestFitness) {
        w->bestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            w->bestPosition[i] = w->position[i];
        }
    }
    if (fitness < *globalBestFitness) {
        *globalBestFitness = fitness;
        for (int i = 0; i < DIMENSIONS; i++) {
            globalBestPosition[i] = w->position[i];
        }
    }
}

void initializeWhales(Whale* whales, double* globalBestPosition, double* globalBestFitness, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < NUM_WHALES; i++) {
        Whale* w = &whales[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            w->position[j] = dist(rng);
            w->bestPosition[j] = w->position[j];
        }
        w->bestFitness = INFINITY;
        updateBestFitness(w, globalBestPosition, globalBestFitness);
    }
}

void updateWhales(Whale* whales, double* globalBestPosition, double* globalBestFitness, std::mt19937& rng, int iter) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < NUM_WHALES; i++) {
        Whale* w = &whales[i];
        double a = 2.0 - (double)iter / MAX_ITERATIONS * 2.0;
        double r1 = dist(rng);
        double r2 = dist(rng);
        double A = 2.0 * a * r1 - a;
        double C = 2.0 * r2;
        double l = dist(rng) * 2.0 - 1.0;
        double p = dist(rng);
        for (int j = 0; j < DIMENSIONS; j++) {
            if (p < 0.5) {
                if (fabs(A) < 1.0) {
                    double D = fabs(C * globalBestPosition[j] - w->position[j]);
                    w->position[j] = globalBestPosition[j] - A * D;
                } else {
                    int randomIndex = (int)(dist(rng) * NUM_WHALES);
                    double D = fabs(C * whales[randomIndex].position[j] - w->position[j]);
                    w->position[j] = whales[randomIndex].position[j] - A * D;
                }
            } else {
                double D = fabs(globalBestPosition[j] - w->position[j]);
                double b = 1.0; // Constant defining the shape of the logarithmic spiral
                w->position[j] = D * exp(b * l) * cos(2.0 * M_PI * l) + globalBestPosition[j];
            }
        }
        updateBestFitness(w, globalBestPosition, globalBestFitness);
    }
}

void runWOA(Whale* whales, double* globalBestPosition, double* globalBestFitness, std::mt19937& rng) {
    std::ofstream outputFile("results.txt");
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateWhales(whales, globalBestPosition, globalBestFitness, rng, iter);

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
    Whale* whales = new Whale[NUM_WHALES];
    double* globalBestPosition = new double[DIMENSIONS];
    double globalBestFitness = INFINITY;

    std::random_device rd;
    std::mt19937 rng(rd());

    auto start = std::chrono::high_resolution_clock::now();
    initializeWhales(whales, globalBestPosition, &globalBestFitness, rng);
    runWOA(whales, globalBestPosition, &globalBestFitness, rng);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(globalBestPosition, globalBestFitness, executionTime);

    delete[] whales;
    delete[] globalBestPosition;

    return 0;
}