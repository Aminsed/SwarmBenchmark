#ifndef PARTICLE_H
#define PARTICLE_H

#include <utility>
#include <cmath>
#include <limits>
#include <functional>

class Particle {
public:
    Particle(double x, double y);
    void updateVelocity(const std::pair<double, double>& globalBestPosition, double inertiaWeight, double cognitiveWeight, double socialWeight, double randP, double randG);
    void updatePosition();
    void evaluateBestPosition(std::function<double(double, double)> objectiveFunc);

    std::pair<double, double> getPosition() const;
    std::pair<double, double> getBestPosition() const;
    double getBestValue() const;

private:
    std::pair<double, double> position;
    std::pair<double, double> velocity{0, 0};
    std::pair<double, double> bestPosition;
    double bestValue = std::numeric_limits<double>::infinity();
};

#endif // PARTICLE_H