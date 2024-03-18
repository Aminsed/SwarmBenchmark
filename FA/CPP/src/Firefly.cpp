#include "Firefly.h"
#include <random>

Firefly::Firefly(double x, double y) : position{x, y} {}

void Firefly::updatePosition(const std::vector<Firefly>& fireflies, double alpha, double betaMin, double gamma, double searchSpaceMin, double searchSpaceMax) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (const auto& other : fireflies) {
        double distance = std::sqrt(std::pow(position.first - other.getPosition().first, 2) +
                                    std::pow(position.second - other.getPosition().second, 2));
        double beta = betaMin * std::exp(-gamma * std::pow(distance, 2));
        position.first += beta * (other.getPosition().first - position.first) + alpha * dis(gen);
        position.second += beta * (other.getPosition().second - position.second) + alpha * dis(gen);

        // Clamp position within search space
        position.first = std::max(searchSpaceMin, std::min(position.first, searchSpaceMax));
        position.second = std::max(searchSpaceMin, std::min(position.second, searchSpaceMax));
    }
}

void Firefly::evaluateIntensity(std::function<double(double, double)> objectiveFunc) {
    intensity = 1.0 / (1.0 + objectiveFunc(position.first, position.second));
}

std::pair<double, double> Firefly::getPosition() const {
    return position;
}

double Firefly::getIntensity() const {
    return intensity;
}
