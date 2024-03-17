#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include <cmath>

class ObjectiveFunction {
public:
    static double rastrigin(double x, double y) {
        return 20 + (x*x - 10*cos(2*M_PI*x)) + (y*y - 10*cos(2*M_PI*y));
    }
};

#endif // OBJECTIVE_FUNCTION_H
