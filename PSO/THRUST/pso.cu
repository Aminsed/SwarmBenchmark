#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>

constexpr int NUM_PARTICLES = 1000;
constexpr int MAX_ITERATIONS = 100;
constexpr double SEARCH_SPACE_MIN = -5.12;
constexpr double SEARCH_SPACE_MAX = 5.12;
constexpr double INERTIA_WEIGHT = 0.729;
constexpr double COGNITIVE_WEIGHT = 1.49;
constexpr double SOCIAL_WEIGHT = 1.49;

struct Particle {
    thrust::pair<double, double> position;
    thrust::pair<double, double> velocity;
    thrust::pair<double, double> bestPosition;
    double bestValue;
};

struct ObjectiveFunction {
    __device__ double operator()(const thrust::pair<double, double>& position) const {
        const double x = position.first;
        const double y = position.second;
        return 20 + x * x - 10 * cos(2 * M_PI * x) + y * y - 10 * cos(2 * M_PI * y);
    }
};

struct UpdateParticle {
    __device__ Particle operator()(Particle particle) {
        ObjectiveFunction objectiveFunction;
        particle.bestValue = min(particle.bestValue, objectiveFunction(particle.position));
        if (particle.bestValue < objectiveFunction(particle.bestPosition)) {
            particle.bestPosition = particle.position;
        }
        return particle;
    }
};

struct CompareParticles {
    __device__ bool operator()(const Particle& p1, const Particle& p2) {
        return p1.bestValue < p2.bestValue;
    }
};

struct UpdateVelocityAndPosition {
    double r1;
    double r2;
    thrust::pair<double, double> globalBestPosition;

    __device__ Particle operator()(Particle particle) {
        particle.velocity.first = INERTIA_WEIGHT * particle.velocity.first +
                                  COGNITIVE_WEIGHT * r1 * (particle.bestPosition.first - particle.position.first) +
                                  SOCIAL_WEIGHT * r2 * (globalBestPosition.first - particle.position.first);
        particle.velocity.second = INERTIA_WEIGHT * particle.velocity.second +
                                   COGNITIVE_WEIGHT * r1 * (particle.bestPosition.second - particle.position.second) +
                                   SOCIAL_WEIGHT * r2 * (globalBestPosition.second - particle.position.second);
        particle.position.first += particle.velocity.first;
        particle.position.second += particle.velocity.second;
        return particle;
    }
};

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(SEARCH_SPACE_MIN, SEARCH_SPACE_MAX);

    thrust::host_vector<Particle> hostSwarm(NUM_PARTICLES);
    for (auto& particle : hostSwarm) {
        particle.position = {dist(gen), dist(gen)};
        particle.velocity = {dist(gen), dist(gen)};
        particle.bestPosition = {0, 0};
        particle.bestValue = std::numeric_limits<double>::max();
    }

    thrust::device_vector<Particle> deviceSwarm = hostSwarm;

    thrust::pair<double, double> globalBestPosition = {0, 0};
    double globalBestValue = std::numeric_limits<double>::max();

    std::uniform_real_distribution<double> r_dist(0, 1);

    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        std::ostringstream oss;
        oss << "particles_" << iteration << ".txt";
        std::string particleFileName = oss.str();
        std::ofstream particleFile(particleFileName);

        thrust::host_vector<Particle> hostSwarmCopy = deviceSwarm;
        for (const auto& particle : hostSwarmCopy) {
            particleFile << particle.position.first << " " << particle.position.second << "\n";
        }
        particleFile << "\n\n";

        thrust::transform(deviceSwarm.begin(), deviceSwarm.end(), deviceSwarm.begin(), UpdateParticle());

        auto minElement = thrust::min_element(deviceSwarm.begin(), deviceSwarm.end(), CompareParticles());
        Particle bestParticle = *minElement;
        if (bestParticle.bestValue < globalBestValue) {
            globalBestPosition = bestParticle.bestPosition;
            globalBestValue = bestParticle.bestValue;
        }

        auto r1 = r_dist(gen);
        auto r2 = r_dist(gen);

        UpdateVelocityAndPosition updateVelocityAndPosition{r1, r2, globalBestPosition};
        thrust::transform(deviceSwarm.begin(), deviceSwarm.end(), deviceSwarm.begin(), updateVelocityAndPosition);
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    std::ofstream bestPositionFile("best_position.txt");
    bestPositionFile << globalBestPosition.first << " " << globalBestPosition.second << "\n";

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Global Best Position: (" << globalBestPosition.first << ", " << globalBestPosition.second << ")\n";
    std::cout << "Global Best Value: " << globalBestValue << "\n";
    std::cout << "Execution Time: " << duration.count() << " milliseconds\n";

    return 0;
}
