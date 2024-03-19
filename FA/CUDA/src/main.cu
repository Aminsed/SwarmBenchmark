#include "FireflySwarm.h"
#include "ObjectiveFunction.h"
#include <iostream>
#include <chrono>

int main() {
    size_t swarmSize = 100;
    double searchSpaceMin = -5.0, searchSpaceMax = 5.0;
    int maxIterations = 1000;

    FireflySwarm swarm(swarmSize, searchSpaceMin, searchSpaceMax, ObjectiveFunction::rosenbrock);

    auto start = std::chrono::high_resolution_clock::now();

    swarm.optimize(maxIterations);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    swarm.printBestFirefly();
    std::cout << "Execution Time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
