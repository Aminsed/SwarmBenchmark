#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

struct Ant {
    int tour[NUM_CITIES];
    double tourLength;
};

__device__ double calculateTourLength(int* tour) {
    double length = 0.0;
    for (int i = 0; i < NUM_CITIES - 1; i++) {
        int city1 = tour[i];
        int city2 = tour[i + 1];
        length += distances[city1][city2];
    }
    int lastCity = tour[NUM_CITIES - 1];
    int firstCity = tour[0];
    length += distances[lastCity][firstCity];
    return length;
}

__global__ void constructTours(Ant* ants, double* pheromones, int* bestTour, double* bestTourLength, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_ANTS) {
        Ant* ant = &ants[tid];
        curandState* s = &state[tid];

        int visited[NUM_CITIES] = {0};
        int current = curand(s) % NUM_CITIES;
        ant->tour[0] = current;
        visited[current] = 1;

        for (int i = 1; i < NUM_CITIES; i++) {
            double probabilities[NUM_CITIES] = {0};
            double sum = 0.0;

            for (int j = 0; j < NUM_CITIES; j++) {
                if (!visited[j]) {
                    probabilities[j] = pow(pheromones[current * NUM_CITIES + j], ALPHA) *
                                       pow(1.0 / distances[current][j], BETA);
                    sum += probabilities[j];
                }
            }

            double r = curand_uniform_double(s) * sum;
            double cumulativeProb = 0.0;
            int nextCity = -1;

            for (int j = 0; j < NUM_CITIES; j++) {
                if (!visited[j]) {
                    cumulativeProb += probabilities[j];
                    if (r <= cumulativeProb) {
                        nextCity = j;
                        break;
                    }
                }
            }

            ant->tour[i] = nextCity;
            visited[nextCity] = 1;
            current = nextCity;
        }

        ant->tourLength = calculateTourLength(ant->tour);

        if (ant->tourLength < *bestTourLength) {
            *bestTourLength = ant->tourLength;
            for (int i = 0; i < NUM_CITIES; i++) {
                bestTour[i] = ant->tour[i];
            }
        }
    }
}

__global__ void updatePheromones(double* pheromones, Ant* ants) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_CITIES * NUM_CITIES) {
        int i = tid / NUM_CITIES;
        int j = tid % NUM_CITIES;
        pheromones[i * NUM_CITIES + j] *= EVAPORATION_RATE;

        for (int k = 0; k < NUM_ANTS; k++) {
            Ant* ant = &ants[k];
            for (int l = 0; l < NUM_CITIES - 1; l++) {
                if (ant->tour[l] == i && ant->tour[l + 1] == j) {
                    pheromones[i * NUM_CITIES + j] += Q / ant->tourLength;
                }
            }
            if (ant->tour[NUM_CITIES - 1] == i && ant->tour[0] == j) {
                pheromones[i * NUM_CITIES + j] += Q / ant->tourLength;
            }
        }
    }
}

void runACO(Ant* ants, double* pheromones, int* bestTour, double* bestTourLength, curandState* state) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_ANTS + block.x - 1) / block.x);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        constructTours<<<grid, block>>>(ants, pheromones, bestTour, bestTourLength, state);
        updatePheromones<<<(NUM_CITIES * NUM_CITIES + block.x - 1) / block.x, block>>>(pheromones, ants);
        cudaDeviceSynchronize();
    }
}

void printResults(int* bestTour, double bestTourLength, double executionTime) {
    std::cout << "Best Tour: ";
    for (int i = 0; i < NUM_CITIES; i++) {
        std::cout << bestTour[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Best Tour Length: " << bestTourLength << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}

int main() {
    Ant* ants;
    double* pheromones;
    int* bestTour;
    double* bestTourLength;
    curandState* state;

    cudaMalloc(&ants, NUM_ANTS * sizeof(Ant));
    cudaMalloc(&pheromones, NUM_CITIES * NUM_CITIES * sizeof(double));
    cudaMalloc(&bestTour, NUM_CITIES * sizeof(int));
    cudaMalloc(&bestTourLength, sizeof(double));
    cudaMalloc(&state, NUM_ANTS * sizeof(curandState));

    double initialPheromone = 1.0;
    cudaMemset(pheromones, initialPheromone, NUM_CITIES * NUM_CITIES * sizeof(double));

    double initialTourLength = INFINITY;
    cudaMemcpy(bestTourLength, &initialTourLength, sizeof(double), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    runACO(ants, pheromones, bestTour, bestTourLength, state);
    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int* hostBestTour = new int[NUM_CITIES];
    double hostBestTourLength;
    cudaMemcpy(hostBestTour, bestTour, NUM_CITIES * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostBestTourLength, bestTourLength, sizeof(double), cudaMemcpyDeviceToHost);

    printResults(hostBestTour, hostBestTourLength, executionTime);

    delete[] hostBestTour;
    cudaFree(ants);
    cudaFree(pheromones);
    cudaFree(bestTour);
    cudaFree(bestTourLength);
    cudaFree(state);
    cudaDeviceReset();

    return 0;
}
