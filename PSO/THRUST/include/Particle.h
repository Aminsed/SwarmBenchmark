// Particle.h
#ifndef PARTICLE_H
#define PARTICLE_H

struct Particle {
    double position[2];
    double velocity[2];
    double bestPosition[2];
    double bestValue;

    __host__ __device__ Particle() : bestValue(1e100) {
        position[0] = position[1] = velocity[0] = velocity[1] = bestPosition[0] = bestPosition[1] = 0.0;
    }
};

#endif // PARTICLE_H
