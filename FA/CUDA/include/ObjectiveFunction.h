#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

class ObjectiveFunction {
public:
    __device__ static double rosenbrock(double x, double y) {
        return 100 * std::pow(y - x * x, 2) + std::pow(1 - x, 2);
    }
};

#endif // OBJECTIVE_FUNCTION_H
