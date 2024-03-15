#ifndef SWARM_H
#define SWARM_H

#include "Particle.h"
#include <vector>
#include <random>

class Swarm {
public:
    Swarm(size_t size, double searchSpaceMin, double searchSpaceMax, double (*objectiveFunc)(double, double));
    void initialize();
    void optimize(int maxIterations, double* d_globalBestPosition);
    double getGlobalBestValue() const;

private:
    std::vector<Particle> particles;
    double globalBestPosition[2];
    double globalBestValue = 1e100;
    double (*objectiveFunc)(double, double);
    double searchSpaceMin, searchSpaceMax;

    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    double randomDouble();
};

#endif // SWARM_H
