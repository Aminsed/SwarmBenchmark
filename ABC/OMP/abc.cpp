#include "ObjectiveFunction.hpp"
#include <omp.h>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

struct FoodSource {
    double position[DIMENSIONS];
    double fitness;
    int trialCount;
};

void updateFitness(FoodSource* fs) {
    fs->fitness = objectiveFunction(fs->position);
}

void initializeFoodSources(FoodSource* foodSources) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-5.0, 5.0);

    #pragma omp parallel for
    for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        FoodSource* fs = &foodSources[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            fs->position[j] = dis(gen);
        }
        updateFitness(fs);
        fs->trialCount = 0;
    }
}

void sendEmployedBees(FoodSource* foodSources) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dimDis(0, DIMENSIONS - 1);
    std::uniform_real_distribution<double> phiDis(-1.0, 1.0);

    #pragma omp parallel for
    for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        FoodSource* fs = &foodSources[i];
        int j = dimDis(gen);
        double phi = phiDis(gen);
        FoodSource newFs = *fs;
        newFs.position[j] += phi * (newFs.position[j] - foodSources[gen() % NUM_FOOD_SOURCES].position[j]);
        updateFitness(&newFs);
        if (newFs.fitness < fs->fitness) {
            *fs = newFs;
            fs->trialCount = 0;
        } else {
            fs->trialCount++;
        }
    }
}

void sendOnlookerBees(FoodSource* foodSources) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::uniform_int_distribution<int> dimDis(0, DIMENSIONS - 1);
    std::uniform_real_distribution<double> phiDis(-1.0, 1.0);

    #pragma omp parallel for
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
        double r = dis(gen) * fitnessSum;
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
        int j = dimDis(gen);
        double phi = phiDis(gen);
        FoodSource newFs = *fs;
        newFs.position[j] += phi * (newFs.position[j] - foodSources[gen() % NUM_FOOD_SOURCES].position[j]);
        updateFitness(&newFs);
        if (newFs.fitness < fs->fitness) {
            *fs = newFs;
            fs->trialCount = 0;
        } else {
            fs->trialCount++;
        }
    }
}

void sendScoutBees(FoodSource* foodSources) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-5.0, 5.0);

    #pragma omp parallel for
    for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
        FoodSource* fs = &foodSources[i];
        if (fs->trialCount >= LIMIT) {
            for (int j = 0; j < DIMENSIONS; j++) {
                fs->position[j] = dis(gen);
            }
            updateFitness(fs);
            fs->trialCount = 0;
        }
    }
}

void runABC(FoodSource* foodSources) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        sendEmployedBees(foodSources);
        sendOnlookerBees(foodSources);
        sendScoutBees(foodSources);
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
    auto start = std::chrono::high_resolution_clock::now();
    initializeFoodSources(foodSources);
    runABC(foodSources);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printResults(foodSources, executionTime);
    return 0;
}
