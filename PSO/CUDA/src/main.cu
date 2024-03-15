#include "Swarm.h"
#include "ObjectiveFunction.h"
#include <iostream>
#include <cuda_runtime.h>

int main() {
    size_t swarmSize = 100;
    double searchSpaceMin = -5.12, searchSpaceMax = 5.12;
    int maxIterations = 1000;

    // Allocate device memory for global best position
    double* d_globalBestPosition;
    cudaMalloc(&d_globalBestPosition, 2 * sizeof(double));

    // Create a swarm with the specified parameters
    Swarm swarm(swarmSize, searchSpaceMin, searchSpaceMax, ObjectiveFunction::rastrigin);

    // Initialize the swarm
    swarm.initialize();

    // Run the PSO algorithm
    swarm.optimize(maxIterations, d_globalBestPosition);

    // Copy global best position from device to host
    double globalBestPosition[2];
    cudaMemcpy(globalBestPosition, d_globalBestPosition, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the global best position and value
    std::cout << "Global Best Position: (" << globalBestPosition[0] << ", " << globalBestPosition[1] << ")" << std::endl;
    std::cout << "Global Best Value: " << swarm.getGlobalBestValue() << std::endl;

    // Free device memory
    cudaFree(d_globalBestPosition);

    return 0;
}
