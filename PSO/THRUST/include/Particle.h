#ifndef PARTICLE_H
#define PARTICLE_H

#include <utility>
#include <cmath>
#include <limits>
#include <functional>
#include <thrust/device_vector.h>

class Particle {
public:
    __host__ __device__ Particle(double x, double y);
    __host__ __device__ void updateVelocity(const thrust::pair<double, double>& globalBestPosition, double inertiaWeight, double cognitiveWeight, double socialWeight, double randP, double randG);
    __host__ __device__ void updatePosition();
    __host__ __device__ void evaluateBestPosition(std::function<double(double, double)> objectiveFunc);

    __host__ __device__ thrust::pair<double, double> getPosition() const;
    __host__ __device__ thrust::pair<double, double> getBestPosition() const;
    __host__ __device__ double getBestValue() const;

private:
    thrust::pair<double, double> position;
    thrust::pair<double, double> velocity{0, 0};
    thrust::pair<double, double> bestPosition;
    double bestValue = thrust::numeric_limits<double>::infinity();
};

#endif // PARTICLE_H
