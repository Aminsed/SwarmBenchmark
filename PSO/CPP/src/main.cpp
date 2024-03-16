#include "Swarm.h"
#include "ObjectiveFunction.h"
#include <iostream>
#include <chrono>

int main() {
    size_t swarmSize = 100;
    double searchSpaceMin = -5.12, searchSpaceMax = 5.12;
    int maxIterations = 1000;

    Swarm swarm(swarmSize, searchSpaceMin, searchSpaceMax, ObjectiveFunction::rastrigin);

    swarm.initialize();

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    swarm.optimize(maxIterations);

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    swarm.printGlobalBest();

    // Print the execution time
    std::cout << "Execution Time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
