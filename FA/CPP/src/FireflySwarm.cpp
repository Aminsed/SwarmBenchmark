#include "FireflySwarm.h"
#include <iostream>
#include <random>
#include <limits>

FireflySwarm::FireflySwarm(size_t size, double searchSpaceMin, double searchSpaceMax, std::function<double(double, double)> objectiveFunc)
    : objectiveFunc(objectiveFunc), searchSpaceMin(searchSpaceMin), searchSpaceMax(searchSpaceMax), dis(searchSpaceMin, searchSpaceMax) {
    std::random_device rd;
    gen.seed(rd());
    for (size_t i = 0; i < size; ++i) {
        fireflies.emplace_back(Firefly(randomDouble(), randomDouble()));
    }
}

void FireflySwarm::optimize(int maxIterations) {
    for (int iter = 0; iter < maxIterations; ++iter) {
        for (auto& firefly : fireflies) {
            firefly.updatePosition(fireflies, alpha, betaMin, gamma, searchSpaceMin, searchSpaceMax);
            firefly.evaluateIntensity(objectiveFunc);
        }
    }
}

void FireflySwarm::printBestFirefly() const {
    auto bestFirefly = std::max_element(fireflies.begin(), fireflies.end(),
                                        [](const Firefly& a, const Firefly& b) {
                                            return a.getIntensity() < b.getIntensity();
                                        });
    std::cout << "Best Firefly Position: (" << bestFirefly->getPosition().first << ", " << bestFirefly->getPosition().second << ")\n";
    std::cout << "Best Firefly Intensity: " << bestFirefly->getIntensity() << std::endl;
}

double FireflySwarm::randomDouble() {
    return dis(gen);
}
