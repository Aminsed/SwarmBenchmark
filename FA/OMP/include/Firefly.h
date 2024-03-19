#ifndef FIREFLY_H
#define FIREFLY_H

#include <utility>
#include <cmath>
#include <vector>
#include <functional>

class Firefly {
public:
    Firefly(double x, double y);
    void updatePosition(const std::vector<Firefly>& fireflies, double alpha, double betaMin, double gamma, double searchSpaceMin, double searchSpaceMax);
    void evaluateIntensity(std::function<double(double, double)> objectiveFunc);

    std::pair<double, double> getPosition() const;
    double getIntensity() const;

private:
    std::pair<double, double> position;
    double intensity;
};

#endif // FIREFLY_H
