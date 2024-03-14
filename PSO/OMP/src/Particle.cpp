#include "Particle.h"

Particle::Particle(double x, double y) : position{x, y}, bestPosition{x, y} {
    bestValue = std::numeric_limits<double>::infinity(); // Ensuring bestValue is properly initialized
}

void Particle::updateVelocity(const std::pair<double, double>& globalBestPosition, double inertiaWeight, double cognitiveWeight, double socialWeight, double randP, double randG) {
    velocity.first = inertiaWeight * velocity.first +
                     cognitiveWeight * randP * (bestPosition.first - position.first) +
                     socialWeight * randG * (globalBestPosition.first - position.first);
    velocity.second = inertiaWeight * velocity.second +
                      cognitiveWeight * randP * (bestPosition.second - position.second) +
                      socialWeight * randG * (globalBestPosition.second - position.second);
}

void Particle::updatePosition() {
    position.first += velocity.first;
    position.second += velocity.second;
}

void Particle::evaluateBestPosition(std::function<double(double, double)> objectiveFunc) {
    double value = objectiveFunc(position.first, position.second);
    if (value < bestValue) {
        bestValue = value;
        bestPosition = position;
    }
}

std::pair<double, double> Particle::getPosition() const {
    return position;
}

std::pair<double, double> Particle::getBestPosition() const {
    return bestPosition;
}

double Particle::getBestValue() const {
    return bestValue;
}