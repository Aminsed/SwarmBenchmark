#include "ObjectiveFunction.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

struct GreyWolf {
    std::vector<double> position;
    double fitness;
};

void initializeGreyWolves(std::vector<GreyWolf>& wolves, std::vector<double>& alpha, std::vector<double>& beta, std::vector<double>& delta) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-5.0, 5.0);

    for (int i = 0; i < NUM_WOLVES; i++) {
        GreyWolf& wolf = wolves[i];
        wolf.position.resize(DIMENSIONS);
        for (int j = 0; j < DIMENSIONS; j++) {
            wolf.position[j] = dis(gen);
        }
        wolf.fitness = objectiveFunction(wolf.position);
        if (wolf.fitness < alpha[DIMENSIONS]) {
            alpha = wolf.position;
            alpha[DIMENSIONS] = wolf.fitness;
        }
        if (wolf.fitness < beta[DIMENSIONS] && wolf.fitness > alpha[DIMENSIONS]) {
            beta = wolf.position;
            beta[DIMENSIONS] = wolf.fitness;
        }
        if (wolf.fitness < delta[DIMENSIONS] && wolf.fitness > beta[DIMENSIONS]) {
            delta = wolf.position;
            delta[DIMENSIONS] = wolf.fitness;
        }
    }
}

void updateGreyWolves(std::vector<GreyWolf>& wolves, std::vector<double>& alpha, std::vector<double>& beta, std::vector<double>& delta, int iter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    double a = 2.0 - (double)iter / MAX_ITERATIONS * 2.0;
    for (int i = 0; i < NUM_WOLVES; i++) {
        GreyWolf& wolf = wolves[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            double r1 = dis(gen);
            double r2 = dis(gen);
            double A1 = 2.0 * a * r1 - a;
            double C1 = 2.0 * r2;
            double D_alpha = std::abs(C1 * alpha[j] - wolf.position[j]);
            double X1 = alpha[j] - A1 * D_alpha;
            r1 = dis(gen);
            r2 = dis(gen);
            double A2 = 2.0 * a * r1 - a;
            double C2 = 2.0 * r2;
            double D_beta = std::abs(C2 * beta[j] - wolf.position[j]);
            double X2 = beta[j] - A2 * D_beta;
            r1 = dis(gen);
            r2 = dis(gen);
            double A3 = 2.0 * a * r1 - a;
            double C3 = 2.0 * r2;
            double D_delta = std::abs(C3 * delta[j] - wolf.position[j]);
            double X3 = delta[j] - A3 * D_delta;
            wolf.position[j] = (X1 + X2 + X3) / 3.0;
        }
        wolf.fitness = objectiveFunction(wolf.position);
        if (wolf.fitness < alpha[DIMENSIONS]) {
            alpha = wolf.position;
            alpha[DIMENSIONS] = wolf.fitness;
        }
        if (wolf.fitness < beta[DIMENSIONS] && wolf.fitness > alpha[DIMENSIONS]) {
            beta = wolf.position;
            beta[DIMENSIONS] = wolf.fitness;
        }
        if (wolf.fitness < delta[DIMENSIONS] && wolf.fitness > beta[DIMENSIONS]) {
            delta = wolf.position;
            delta[DIMENSIONS] = wolf.fitness;
        }
    }
}

void runGWO(std::vector<GreyWolf>& wolves, std::vector<double>& alpha, std::vector<double>& beta, std::vector<double>& delta) {
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateGreyWolves(wolves, alpha, beta, delta, iter);
    }
}

void printResults(const std::vector<double>& alpha, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Best Position: " << alpha[0] << std::endl;
    } else {
        std::cout << "Best Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << alpha[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Fitness: " << alpha[DIMENSIONS] << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    std::vector<GreyWolf> wolves(NUM_WOLVES);
    std::vector<double> alpha(DIMENSIONS + 1, INFINITY);
    std::vector<double> beta(DIMENSIONS + 1, INFINITY);
    std::vector<double> delta(DIMENSIONS + 1, INFINITY);

    auto start = std::chrono::high_resolution_clock::now();
    initializeGreyWolves(wolves, alpha, beta, delta);
    runGWO(wolves, alpha, beta, delta);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printResults(alpha, executionTime);

    return 0;
}
