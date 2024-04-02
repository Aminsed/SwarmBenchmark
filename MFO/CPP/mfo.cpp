#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>

struct Moth {
    double position[DIMENSIONS];
    double fitness;
};

struct Flame {
    double position[DIMENSIONS];
};

void updateFlame(Moth& m, Flame& flame, double& bestFitness) {
    double fitness = objectiveFunction(m.position);
    if (fitness < objectiveFunction(flame.position)) {
        for (int i = 0; i < DIMENSIONS; i++) {
            flame.position[i] = m.position[i];
        }
    }
    if (fitness < bestFitness) {
        bestFitness = fitness;
    }
}

void initializeMoths(std::vector<Moth>& moths, std::vector<Flame>& flames, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < NUM_MOTHS; i++) {
        Moth& m = moths[i];
        Flame& f = flames[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            m.position[j] = dist(rng);
            f.position[j] = m.position[j];
        }
        m.fitness = objectiveFunction(m.position);
    }
}

void updateMoths(std::vector<Moth>& moths, std::vector<Flame>& flames, std::vector<int>& flameIndexes, std::mt19937& rng, int iter, double& bestFitness) {
    for (int i = 0; i < NUM_MOTHS; i++) {
        std::mt19937 moth_rng(rng());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        Moth& m = moths[i];
        int flameIndex = flameIndexes[i];
        Flame& flame = flames[flameIndex];
        for (int j = 0; j < DIMENSIONS; j++) {
            double t = static_cast<double>(iter) / MAX_ITERATIONS;
            double r = dist(moth_rng);
            double b = 1.0;
            double distance = std::abs(flame.position[j] - m.position[j]);
            if (r < 0.5) {
                m.position[j] = distance * std::exp(b * t) * std::cos(t * 2 * M_PI) + flame.position[j];
            } else {
                m.position[j] = distance * std::exp(b * t) * std::sin(t * 2 * M_PI) + flame.position[j];
            }
        }
        m.fitness = objectiveFunction(m.position);
        updateFlame(m, flame, bestFitness);
    }
}

void sortMothsByFitness(std::vector<Moth>& moths, std::vector<int>& flameIndexes) {
    std::sort(flameIndexes.begin(), flameIndexes.end(), [&](int i, int j) {
        return moths[i].fitness < moths[j].fitness;
    });
}

void runMFO(std::vector<Moth>& moths, std::vector<Flame>& flames, std::vector<int>& flameIndexes, std::mt19937& rng, double& bestFitness) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateMoths(moths, flames, flameIndexes, rng, iter, bestFitness);
        sortMothsByFitness(moths, flameIndexes);
    }
}

void printResults(const std::vector<Flame>& flames, double bestFitness, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Best Flame Position: " << flames[0].position[0] << std::endl;
    } else {
        std::cout << "Best Flame Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << flames[0].position[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Flame Fitness: " << bestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    std::vector<Moth> moths(NUM_MOTHS);
    std::vector<Flame> flames(NUM_MOTHS);
    std::vector<int> flameIndexes(NUM_MOTHS);
    std::mt19937 rng(std::random_device{}());
    double bestFitness = std::numeric_limits<double>::max();

    auto start = std::chrono::high_resolution_clock::now();
    initializeMoths(moths, flames, rng);
    runMFO(moths, flames, flameIndexes, rng, bestFitness);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(flames, bestFitness, executionTime);

    return 0;
}
