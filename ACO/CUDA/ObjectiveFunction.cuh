#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define NUM_ANTS 1024
#define MAX_ITERATIONS 1000
#define PHEROMONE_WEIGHT 0.8 // Weight of pheromone in the probability calculation
#define Q 100.0 // Constant for pheromone update

/*
// Rosenbrock function
#define DIMENSIONS 1 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double x = position[0];
    return 100.0 * (x * x - x) * (x * x - x) + (1.0 - x) * (1.0 - x);
}
*/
/*
// Rastrigin function
#define DIMENSIONS 2 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double sum = 0.0;
    for (int i = 0; i < DIMENSIONS; i++) {
        double xi = position[i];
        sum += (xi * xi - 10.0 * cos(2.0 * M_PI * xi));
    }
    return 20.0 + sum;
}
*/

/*
// Griewank function
#define DIMENSIONS 3 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double x = position[0];
    double y = position[1];
    double z = position[2];
    return 1.0 + (x * x + y * y + z * z) / 4000.0 - cos(x) * cos(y / sqrt(2.0)) * cos(z / sqrt(3.0));
}
*/

/*
// Schaffer function N.4
#define DIMENSIONS 4 // Number of dimensions in the optimization problem
__device__ double objectiveFunction(double* position) {
    double x = position[0];
    double y = position[1];
    double numerator = pow(cos(sin(fabs(x * x - y * y))), 2) - 0.5;
    double denominator = pow(1.0 + 0.001 * (x * x + y * y), 2);
    return 0.5 + numerator / denominator;
}
*/

// TSP objective function
#define DIMENSIONS 10 // Number of cities in the TSP problem
__device__ double distance[DIMENSIONS][DIMENSIONS] = {
    {0.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0},
    {3.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0},
    {4.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0},
    {5.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {1.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0},
    {2.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0},
    {3.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0},
    {4.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0},
    {5.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0}
};

struct City {
    int index;
    double value;
};

__device__ void sortCities(City* cities, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (cities[j].value > cities[j + 1].value) {
                // Swap cities
                City temp = cities[j];
                cities[j] = cities[j + 1];
                cities[j + 1] = temp;
            }        }
    }
}

__device__ double objectiveFunction(double* position) {
    double totalDistance = 0.0;
    int* tour = new int[DIMENSIONS];
    City* cities = new City[DIMENSIONS];
    
    // Convert position to cities
    for (int i = 0; i < DIMENSIONS; i++) {
        cities[i].index = i;
        cities[i].value = position[i];
    }
    
    // Sort cities based on position values
    sortCities(cities, DIMENSIONS);
    
    // Create tour based on sorted cities
    for (int i = 0; i < DIMENSIONS; i++) {
        tour[i] = cities[i].index;
    }
    
    // Calculate total distance
    for (int i = 0; i < DIMENSIONS - 1; i++) {
        int city1 = tour[i];
        int city2 = tour[i + 1];
        totalDistance += distance[city1][city2];
    }
    int lastCity = tour[DIMENSIONS - 1];
    int firstCity = tour[0];
    totalDistance += distance[lastCity][firstCity];
    
    delete[] tour;
    delete[] cities;
    return totalDistance;
}

#endif
