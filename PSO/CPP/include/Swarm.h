#ifndef SWARM_H
#define SWARM_H

#include "Particle.h"
#include <vector>
#include <random>
#include <functional>

class Swarm {
public:
    Swarm(size_t size, double searchSpaceMin, double searchSpaceMax, std::function<double(double, double)> objectiveFunc);
    void initialize();
    void optimize(int maxIterations);
    void printGlobalBest() const;

private:
    std::vector<Particle> particles;
    std::pair<double, double> globalBestPosition;
    double globalBestValue = std::numeric_limits<double>::infinity();
    std::function<double(double, double)> objectiveFunc;
    double searchSpaceMin, searchSpaceMax;

    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    void updateGlobalBest();
    double randomDouble();
};

#endif // SWARM_H