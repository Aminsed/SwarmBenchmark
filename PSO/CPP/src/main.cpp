#include "Swarm.h"
#include "ObjectiveFunction.h"

int main() {
    size_t swarmSize = 100;
    double searchSpaceMin = -5.12, searchSpaceMax = 5.12;
    int maxIterations = 1000;
    Swarm swarm(swarmSize, searchSpaceMin, searchSpaceMax, ObjectiveFunction::rastrigin);

    swarm.initialize();
    swarm.optimize(maxIterations);
    swarm.printGlobalBest();

    return 0;
}