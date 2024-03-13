// pso.cpp
// Implementation of the Particle Swarm Optimization algorithm
// for finding the global minimum of the Rastrigin function.

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>

const int NUM_PARTICLES = 1000;
const int MAX_ITERATIONS = 100;
const double INERTIA_WEIGHT = 0.729;
const double COGNITIVE_WEIGHT = 1.49;
const double SOCIAL_WEIGHT = 1.49;

struct Particle {
    std::pair<double, double> position;
    std::pair<double, double> velocity;
    std::pair<double, double> bestPosition;
    double bestValue;
};

class ObjectiveFunction {
public:
    static double calculate(const std::pair<double, double>& position) {
        const auto& [positionX, positionY] = position;
        return 20 + positionX * positionX - 10 * std::cos(2 * M_PI * positionX)
               + positionY * positionY - 10 * std::cos(2 * M_PI * positionY);
    }
};

class SwarmInitializer {
public:
    static void initialize(std::vector<Particle>& swarm, std::mt19937& generator,
                           std::uniform_real_distribution<double>& positionDist,
                           std::uniform_real_distribution<double>& velocityDist) {
        for (auto& particle : swarm) {
            initializeParticle(particle, generator, positionDist, velocityDist);
        }
    }

private:
    static void initializeParticle(Particle& particle, std::mt19937& generator,
                                   std::uniform_real_distribution<double>& positionDist,
                                   std::uniform_real_distribution<double>& velocityDist) {
        particle.position = {positionDist(generator), positionDist(generator)};
        particle.velocity = {velocityDist(generator), velocityDist(generator)};
        particle.bestPosition = particle.position;
        particle.bestValue = ObjectiveFunction::calculate(particle.position);
    }
};

class ParticleUpdater {
public:
    static void updatePersonalBest(Particle& particle) {
        const double currentValue = ObjectiveFunction::calculate(particle.position);
        if (currentValue < particle.bestValue) {
            particle.bestValue = currentValue;
            particle.bestPosition = particle.position;
        }
    }

    static void updateGlobalBest(const Particle& particle, std::pair<double, double>& globalBestPosition,
                                 double& globalBestValue) {
        if (particle.bestValue < globalBestValue) {
            globalBestPosition = particle.bestPosition;
            globalBestValue = particle.bestValue;
        }
    }

    static void updateVelocityAndPosition(Particle& particle, std::mt19937& generator,
                                          std::uniform_real_distribution<double>& randomDist,
                                          const std::pair<double, double>& globalBestPosition) {
        updateVelocityComponent(particle.velocity.first, particle.bestPosition.first,
                                globalBestPosition.first, particle.position.first, randomDist(generator));
        updateVelocityComponent(particle.velocity.second, particle.bestPosition.second,
                                globalBestPosition.second, particle.position.second, randomDist(generator));
        particle.position.first += particle.velocity.first;
        particle.position.second += particle.velocity.second;
    }

private:
    static void updateVelocityComponent(double& velocity, const double personalBest, const double globalBest,
                                        const double currentPosition, const double randomValue) {
        velocity = INERTIA_WEIGHT * velocity
                 + COGNITIVE_WEIGHT * randomValue * (personalBest - currentPosition)
                 + SOCIAL_WEIGHT * randomValue * (globalBest - currentPosition);
    }
};

class FileWriter {
public:
    static void writeParticlePositions(const std::vector<Particle>& swarm, const int iteration) {
        const std::string filename = "particles_" + std::to_string(iteration) + ".txt";
        writeToFile(filename, [&](std::ofstream& file) {
            for (const auto& particle : swarm) {
                file << particle.position.first << " " << particle.position.second << "\n";
            }
            file << "\n";
        });
    }

    static void writeGlobalBest(const std::pair<double, double>& globalBestPosition) {
        writeToFile("best_position.txt", [&](std::ofstream& file) {
            file << globalBestPosition.first << " " << globalBestPosition.second << "\n";
        });
    }

private:
    template <typename FileWriterCallback>
    static void writeToFile(const std::string& filename, FileWriterCallback writerCallback) {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        writerCallback(file);
    }
};

class ResultPrinter {
public:
    static void printResults(const std::pair<double, double>& globalBestPosition, const double globalBestValue,
                             const std::chrono::milliseconds& duration) {
        std::cout << "Global Best Position: (" << globalBestPosition.first << ", "
                  << globalBestPosition.second << ")\n";
        std::cout << "Global Best Value:    " << globalBestValue << "\n";
        std::cout << "Execution Time:       " << duration.count() << " milliseconds\n";
    }
};

class ParticleSwarmOptimization {
private:
    std::vector<Particle> swarm;
    std::pair<double, double> globalBestPosition;
    double globalBestValue;

public:
    ParticleSwarmOptimization() : swarm(NUM_PARTICLES) {
        globalBestPosition = swarm[0].bestPosition;
        globalBestValue = swarm[0].bestValue;
    }

    void optimize() {
        const auto startTime = std::chrono::high_resolution_clock::now();

        initializeSwarm();
        runOptimization();

        const auto endTime = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        FileWriter::writeGlobalBest(globalBestPosition);
        ResultPrinter::printResults(globalBestPosition, globalBestValue, duration);
    }

private:
    void initializeSwarm() {
        std::random_device randomDevice;
        std::mt19937 generator(randomDevice());
        const double SEARCH_SPACE_MIN = -5.12;
        const double SEARCH_SPACE_MAX = 5.12;
        std::uniform_real_distribution<double> positionDist(SEARCH_SPACE_MIN, SEARCH_SPACE_MAX);
        std::uniform_real_distribution<double> velocityDist(SEARCH_SPACE_MIN, SEARCH_SPACE_MAX);

        SwarmInitializer::initialize(swarm, generator, positionDist, velocityDist);
    }

    void runOptimization() {
        std::random_device randomDevice;
        std::mt19937 generator(randomDevice());
        std::uniform_real_distribution<double> randomDist(0, 1);

        for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
            FileWriter::writeParticlePositions(swarm, iteration);
            updateParticles(generator, randomDist);
        }
    }

    void updateParticles(std::mt19937& generator, std::uniform_real_distribution<double>& randomDist) {
        for (auto& particle : swarm) {
            updateParticle(particle, generator, randomDist);
        }
    }

    void updateParticle(Particle& particle, std::mt19937& generator,
                        std::uniform_real_distribution<double>& randomDist) {
        ParticleUpdater::updatePersonalBest(particle);
        ParticleUpdater::updateGlobalBest(particle, globalBestPosition, globalBestValue);
        ParticleUpdater::updateVelocityAndPosition(particle, generator, randomDist, globalBestPosition);
    }
};

void runOptimization() {
    try {
        ParticleSwarmOptimization pso;
        pso.optimize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        throw;
    }
}

int main() {
    try {
        runOptimization();
    } catch (...) {
        return 1;
    }

    return 0;
}