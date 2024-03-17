#include "Particle.h"

__host__ __device__ Particle::Particle(double x, double y) : position{x, y}, bestPosition{x, y} {
    bestValue = thrust::numeric_limits<double>::infinity();
}

__host__ __device__ void Particle::updateVelocity(const thrust::pair<double, double>& globalBestPosition, double inertiaWeight, double cognitiveWeight, double socialWeight, double randP, double randG) {
    velocity.first = inertiaWeight * velocity.first +
                     cognitiveWeight * randP * (bestPosition.first - position.first) +
                     socialWeight * randG * (globalBestPosition.first - position.first);
    velocity.second = inertiaWeight * velocity.second +
                      cognitiveWeight * randP * (bestPosition.second - position.second) +
                      socialWeight * randG * (globalBestPosition.second - position.second);
}

__host__ __device__ void Particle::updatePosition() {
    position.first += velocity.first;
    position.second += velocity.second;
}

__host__ __device__ void Particle::evaluateBestPosition(std::function<double(double, double)> objectiveFunc) {
    double value = objectiveFunc(position.first, position.second);
    if (value < bestValue) {
        bestValue = value;
        bestPosition = position;
    }
}

__host__ __device__ thrust::pair<double, double> Particle::getPosition() const {
    return position;
}

__host__ __device__ thrust::pair<double, double> Particle::getBestPosition() const {
    return bestPosition;
}

__host__ __device__ double Particle::getBestValue() const {
    return bestValue;
}
