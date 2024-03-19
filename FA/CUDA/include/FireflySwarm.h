#ifndef FIREFLY_SWARM_H
#define FIREFLY_SWARM_H

#include "Firefly.h"
#include <vector>
#include <functional>
#include <random>

class FireflySwarm {
public:
    FireflySwarm(size_t size, double searchSpaceMin, double searchSpaceMax, std::function<double(double, double)> objectiveFunc);
    ~FireflySwarm();
    void optimize(int maxIterations);
    void printBestFirefly() const;

private:
    Firefly* fireflies;
    Firefly* d_fireflies;
    size_t numFireflies;
    std::function<double(double, double)> objectiveFunc;
    double searchSpaceMin, searchSpaceMax;
    double alpha = 0.2, betaMin = 0.2, gamma = 1.0;

    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

    double randomDouble();
};

#endif // FIREFLY_SWARM_H
