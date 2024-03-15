#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>
#include <cmath>

class Particle {
public:
    __device__ Particle() {
        position[0] = 0.0;
        position[1] = 0.0;
        bestPosition[0] = 0.0;
        bestPosition[1] = 0.0;
        bestValue = 1e100; // Initialize bestValue with a large finite value
    }

    __device__ Particle(double x, double y) {
        position[0] = x;
        position[1] = y;
        bestPosition[0] = x;
        bestPosition[1] = y;
        bestValue = 1e100; // Initialize bestValue with a large finite value
    }

    __device__ void updateVelocity(const double* globalBestPosition, double inertiaWeight, double cognitiveWeight, double socialWeight, double randP, double randG) {
        velocity[0] = inertiaWeight * velocity[0] +
                      cognitiveWeight * randP * (bestPosition[0] - position[0]) +
                      socialWeight * randG * (globalBestPosition[0] - position[0]);
        velocity[1] = inertiaWeight * velocity[1] +
                      cognitiveWeight * randP * (bestPosition[1] - position[1]) +
                      socialWeight * randG * (globalBestPosition[1] - position[1]);
    }

    __device__ void updatePosition() {
        position[0] += velocity[0];
        position[1] += velocity[1];
    }

    __device__ void evaluateBestPosition(double (*objectiveFunc)(double, double)) {
        double value = objectiveFunc(position[0], position[1]);
        if (value < bestValue) {
            bestValue = value;
            bestPosition[0] = position[0];
            bestPosition[1] = position[1];
        }
    }

    __device__ const double* getPosition() const {
        return position;
    }

    __device__ const double* getBestPosition() const {
        return bestPosition;
    }

    __device__ double getBestValue() const {
        return bestValue;
    }

private:
    double position[2];
    double velocity[2] = {0, 0};
    double bestPosition[2];
    double bestValue = 1e100; // Initialize bestValue with a large finite value
};

#endif // PARTICLE_H
