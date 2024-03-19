#ifndef FIREFLY_H
#define FIREFLY_H

#include <utility>
#include <cmath>
#include <curand_kernel.h>

__device__ inline double deviceRandomDouble(double min, double max, curandState* state) {
    return curand_uniform_double(state) * (max - min) + min;
}

class Firefly {
public:
    __host__ __device__ Firefly() : position{0.0, 0.0}, intensity(0.0) {}
    __host__ __device__ Firefly(double x, double y) : position{x, y}, intensity(0.0) {}

    __device__ void updatePosition(const Firefly* fireflies, size_t numFireflies, double alpha, double betaMin, double gamma, double searchSpaceMin, double searchSpaceMax) {
        curandState localState;
        curand_init(clock64(), threadIdx.x, 0, &localState);

        for (size_t i = 0; i < numFireflies; ++i) {
            const Firefly& other = fireflies[i];
            double distance = std::sqrt(std::pow(position.first - other.getPosition().first, 2) +
                                        std::pow(position.second - other.getPosition().second, 2));
            double beta = betaMin * std::exp(-gamma * std::pow(distance, 2));
            position.first += beta * (other.getPosition().first - position.first) + alpha * deviceRandomDouble(-1.0, 1.0, &localState);
            position.second += beta * (other.getPosition().second - position.second) + alpha * deviceRandomDouble(-1.0, 1.0, &localState);

            // Clamp position within search space
            position.first = max(searchSpaceMin, min(position.first, searchSpaceMax));
            position.second = max(searchSpaceMin, min(position.second, searchSpaceMax));
        }
    }

    __device__ void evaluateIntensity(double (*objectiveFunc)(double, double)) {
        intensity = 1.0 / (1.0 + objectiveFunc(position.first, position.second));
    }

    __device__ std::pair<double, double> getPosition() const {
        return position;
    }

    __device__ double getIntensity() const {
        return intensity;
    }

    __host__ std::pair<double, double> getPositionHost() const {
        return position;
    }

    __host__ double getIntensityHost() const {
        return intensity;
    }

private:
    std::pair<double, double> position;
    double intensity;
};

#endif // FIREFLY_H
