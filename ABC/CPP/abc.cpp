#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

struct FoodSource {
    double position[DIMENSIONS];
    double fitness;
    int trialCount;
};

void updateFitness(FoodSource* fs) {
    fs->fitness = objectiveFunction(fs->position);
}

void initializeFoodSources(FoodSource* foodSources, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        FoodSource* fs = &foodSources[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            fs->position[j] = dist(rng);
        }
        updateFitness(fs);
        fs->trialCount = 0;
    }
}

void sendEmployedBees(FoodSource* foodSources, std::mt19937& rng) {
    std::uniform_int_distribution<int> dimDist(0, DIMENSIONS - 1);
    std::uniform_real_distribution<double> phiDist(-1.0, 1.0);
    std::uniform_int_distribution<int> fsDist(0, NUM_FOOD_SOURCES - 1);
    for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        FoodSource* fs = &foodSources[i];
        int j = dimDist(rng);
        double phi = phiDist(rng);
        FoodSource newFs = *fs;
        newFs.position[j] += phi * (newFs.position[j] - foodSources[fsDist(rng)].position[j]);
        updateFitness(&newFs);
        if (newFs.fitness < fs->fitness) {
            *fs = newFs;
            fs->trialCount = 0;
        } else {
            fs->trialCount++;
        }
    }
}

void sendOnlookerBees(FoodSource* foodSources, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> dimDist(0, DIMENSIONS - 1);
    std::uniform_real_distribution<double> phiDist(-1.0, 1.0);
    std::uniform_int_distribution<int> fsDist(0, NUM_FOOD_SOURCES - 1);
    for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        double probabilities[NUM_FOOD_SOURCES];
        double maxFitness = foodSources[0].fitness;
        for (int j = 1; j < NUM_FOOD_SOURCES; j++) {
            if (foodSources[j].fitness > maxFitness) {
                maxFitness = foodSources[j].fitness;
            }
        }
        double fitnessSum = 0.0;
        for (int j = 0; j < NUM_FOOD_SOURCES; j++) {
            probabilities[j] = (0.9 * (foodSources[j].fitness / maxFitness)) + 0.1;
            fitnessSum += probabilities[j];
        }
        double r = dist(rng) * fitnessSum;
        double cumulativeProbability = 0.0;
        int selectedIndex = 0;
        for (int j = 0; j < NUM_FOOD_SOURCES; j++) {
            cumulativeProbability += probabilities[j];
            if (r <= cumulativeProbability) {
                selectedIndex = j;
                break;
            }
        }
        FoodSource* fs = &foodSources[selectedIndex];
        int j = dimDist(rng);
        double phi = phiDist(rng);
        FoodSource newFs = *fs;
        newFs.position[j] += phi * (newFs.position[j] - foodSources[fsDist(rng)].position[j]);
        updateFitness(&newFs);
        if (newFs.fitness < fs->fitness) {
            *fs = newFs;
            fs->trialCount = 0;
        } else {
            fs->trialCount++;
        }
    }
}

void sendScoutBees(FoodSource* foodSources, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        FoodSource* fs = &foodSources[i];
        if (fs->trialCount >= LIMIT) {
            for (int j = 0; j < DIMENSIONS; j++) {
                fs->position[j] = dist(rng);
            }
            updateFitness(fs);
            fs->trialCount = 0;
        }
    }
}

void runABC(FoodSource* foodSources, std::mt19937& rng) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        sendEmployedBees(foodSources, rng);
        sendOnlookerBees(foodSources, rng);
        sendScoutBees(foodSources, rng);
    }
}

void printResults(FoodSource* foodSources, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    FoodSource bestFoodSource = foodSources[0];
    for (int i = 1; i < NUM_FOOD_SOURCES; i++) {
        if (foodSources[i].fitness < bestFoodSource.fitness) {
            bestFoodSource = foodSources[i];
        }
    }
    if (DIMENSIONS == 1) {
        std::cout << "Best Food Source Position: " << bestFoodSource.position[0] << std::endl;
    } else {
        std::cout << "Best Food Source Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << bestFoodSource.position[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Food Source Fitness: " << bestFoodSource.fitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    FoodSource foodSources[NUM_FOOD_SOURCES];
    std::random_device rd;
    std::mt19937 rng(rd());
    auto start = std::chrono::high_resolution_clock::now();
    initializeFoodSources(foodSources, rng);
    runABC(foodSources, rng);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printResults(foodSources, executionTime);
    return 0;
}
