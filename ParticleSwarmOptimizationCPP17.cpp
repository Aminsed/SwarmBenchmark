#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

constexpr int NUM_PARTICLES = 1000;
constexpr int MAX_ITERATIONS = 100;
constexpr double SEARCH_SPACE_MIN = -5.12;
constexpr double SEARCH_SPACE_MAX = 5.12;
constexpr double INERTIA_WEIGHT = 0.729;
constexpr double COGNITIVE_WEIGHT = 1.49;
constexpr double SOCIAL_WEIGHT = 1.49;

struct Particle {
    std::pair<double, double> position;
    std::pair<double, double> velocity;
    std::pair<double, double> bestPosition;
    double bestValue;
};

double objectiveFunction(const std::pair<double, double>& position) {
    const auto& [x, y] = position;
    return 20 + x * x - 10 * std::cos(2 * M_PI * x) + y * y - 10 * std::cos(2 * M_PI * y);
}

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(SEARCH_SPACE_MIN, SEARCH_SPACE_MAX);

    std::vector<Particle> swarm(NUM_PARTICLES);
    for (auto& particle : swarm) {
        particle.position = {dist(gen), dist(gen)};
        particle.velocity = {dist(gen), dist(gen)};
        particle.bestPosition = {0, 0};
        particle.bestValue = std::numeric_limits<double>::max();
    }

    std::pair<double, double> globalBestPosition = {0, 0};
    double globalBestValue = std::numeric_limits<double>::max();

    std::uniform_real_distribution<double> r_dist(0, 1);
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        std::ostringstream oss;
        oss << "particles_" << iteration << ".txt";
        std::string particleFileName = oss.str();
        std::ofstream particleFile(particleFileName);

        for (const auto& particle : swarm) {
            particleFile << particle.position.first << " " << particle.position.second << "\n";
        }
        particleFile << "\n\n";

        for (auto& particle : swarm) {
            particle.bestValue = std::min(particle.bestValue, objectiveFunction(particle.position));
            if (particle.bestValue < objectiveFunction(particle.bestPosition)) {
                particle.bestPosition = particle.position;
            }
            if (particle.bestValue < globalBestValue) {
                globalBestPosition = particle.bestPosition;
                globalBestValue = particle.bestValue;
            }

            auto r1 = r_dist(gen);
            auto r2 = r_dist(gen);
            particle.velocity.first = INERTIA_WEIGHT * particle.velocity.first +
                                      COGNITIVE_WEIGHT * r1 * (particle.bestPosition.first - particle.position.first) +
                                      SOCIAL_WEIGHT * r2 * (globalBestPosition.first - particle.position.first);
            particle.velocity.second = INERTIA_WEIGHT * particle.velocity.second +
                                       COGNITIVE_WEIGHT * r1 * (particle.bestPosition.second - particle.position.second) +
                                       SOCIAL_WEIGHT * r2 * (globalBestPosition.second - particle.position.second);

            particle.position.first += particle.velocity.first;
            particle.position.second += particle.velocity.second;
        }
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
