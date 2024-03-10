#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

using namespace std;

const double alpha = 1.0; // pheromone importance
const double betaValue = 5.0; // distance priority
const double evaporation = 0.5;
const double Q = 100; // pheromone left on trail per ant
const double antFactor = 0.8; // No. of ants per city

int numberOfCities = 10;
int numberOfAnts = 5;
vector<vector<double>> distances;
vector<vector<double>> pheromones;

vector<int> bestTour;
double bestTourLength = numeric_limits<double>::max();

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void evaporatePheromonesKernel(double* pheromones, int numberOfCities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numberOfCities && j < numberOfCities) {
        pheromones[i * numberOfCities + j] *= evaporation;
    }
}


__global__ void updatePheromonesKernel(double* pheromones, int* tour, double tourLength, int numberOfCities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numberOfCities - 1) {
        int city1 = tour[i];
        int city2 = tour[i + 1];
        if (city1 >= 0 && city1 < numberOfCities && city2 >= 0 && city2 < numberOfCities) {
            double contribution = Q / tourLength;
            printf("Updating pheromone for edge (%d, %d) with contribution %f\n", city1, city2, contribution);
            pheromones[city1 * numberOfCities + city2] += contribution;
        } else {
            printf("Invalid city indices: city1=%d, city2=%d\n", city1, city2);
        }
    }

    if (i == 0) {
        int city1 = tour[numberOfCities - 1];
        int city2 = tour[0];
        if (city1 >= 0 && city1 < numberOfCities && city2 >= 0 && city2 < numberOfCities) {
            double contribution = Q / tourLength;
            printf("Updating pheromone for edge (%d, %d) with contribution %f\n", city1, city2, contribution);
            pheromones[city1 * numberOfCities + city2] += contribution;
        } else {
            printf("Invalid city indices: city1=%d, city2=%d\n", city1, city2);
        }
    }
}



void initialize() {
    distances.resize(numberOfCities, vector<double>(numberOfCities));
    pheromones.resize(numberOfCities, vector<double>(numberOfCities, 1));

    srand(time(NULL)); // seed for random number generator
    for (int i = 0; i < numberOfCities; i++) {
        for (int j = i + 1; j < numberOfCities; j++) {
            double dist = rand() % 100 + 1; // distance between cities
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    cout << "Distances matrix:" << endl;
    for (int i = 0; i < numberOfCities; i++) {
        for (int j = 0; j < numberOfCities; j++) {
            cout << distances[i][j] << " ";
        }
        cout << endl;
    }
}

void updatePheromones(double* d_pheromones, int* d_tours, double* d_lengths) {
    dim3 blockSize(16, 16);
    dim3 gridSize((numberOfCities + blockSize.x - 1) / blockSize.x, (numberOfCities + blockSize.y - 1) / blockSize.y);

    cout << "Evaporating pheromones..." << endl;
    evaporatePheromonesKernel<<<gridSize, blockSize>>>(d_pheromones, numberOfCities);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < numberOfAnts; i++) {
        cout << "Updating pheromones for ant " << i << endl;
        updatePheromonesKernel<<<(numberOfCities + 255) / 256, 256>>>(d_pheromones, d_tours + i * numberOfCities, d_lengths[i], numberOfCities);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

double calculateProbability(int from, int to, bool* visited) {
    double pheromone = pow(pheromones[from][to], alpha);
    double inverseDistance = pow(1.0 / distances[from][to], betaValue);
    if (visited[to]) return 0.0;
    else return pheromone * inverseDistance;
}

int selectNextCity(int currentCity, bool* visited) {
    vector<double> probabilities(numberOfCities);
    double sumProbabilities = 0.0;

    for (int i = 0; i < numberOfCities; i++) {
        if (!visited[i]) {
            probabilities[i] = calculateProbability(currentCity, i, visited);
            sumProbabilities += probabilities[i];
        }
    }

    if (sumProbabilities == 0) return -1; // No unvisited cities

    double random = rand() / (double)RAND_MAX;
    double cumulativeProbability = 0.0;
    for (int i = 0; i < numberOfCities; i++) {
        if (!visited[i]) {
            probabilities[i] /= sumProbabilities;
            cumulativeProbability += probabilities[i];
            if (random <= cumulativeProbability) {
                return i;
            }
        }
    }

    return -1; // should not reach here
}

void findBestTour() {
    size_t availableMemory, totalMemory;
    CUDA_CHECK(cudaMemGetInfo(&availableMemory, &totalMemory));
    cout << "Available GPU memory: " << availableMemory << " bytes" << endl;
    cout << "Total GPU memory: " << totalMemory << " bytes" << endl;

    double* d_pheromones;
    int* d_tours;
    double* d_lengths;

    size_t pheromonesSize = numberOfCities * numberOfCities * sizeof(double);
    size_t toursSize = numberOfAnts * numberOfCities * sizeof(int);
    size_t lengthsSize = numberOfAnts * sizeof(double);

    cout << "Pheromones memory size: " << pheromonesSize << " bytes" << endl;
    cout << "Tours memory size: " << toursSize << " bytes" << endl;
    cout << "Lengths memory size: " << lengthsSize << " bytes" << endl;

    if (pheromonesSize + toursSize + lengthsSize > availableMemory) {
        cout << "Insufficient GPU memory!" << endl;
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMalloc(&d_pheromones, pheromonesSize));
    CUDA_CHECK(cudaMalloc(&d_tours, toursSize));
    CUDA_CHECK(cudaMalloc(&d_lengths, lengthsSize));

    for (int iteration = 0; iteration < 100; iteration++) {
        cout << "Iteration " << iteration << endl;

        vector<vector<int>> tours(numberOfAnts, vector<int>(numberOfCities));
        vector<double> lengths(numberOfAnts, 0.0);

        for (int i = 0; i < numberOfAnts; i++) {
            bool visited[numberOfCities] = {false}; // Use a regular boolean array
            tours[i][0] = rand() % numberOfCities;
            visited[tours[i][0]] = true;

            cout << "Ant " << i << " starting city: " << tours[i][0] << endl;

            for (int j = 1; j < numberOfCities; j++) {
                int nextCity = selectNextCity(tours[i][j - 1], visited);
                if (nextCity == -1) {
                    cout << "Error: No unvisited city found for ant " << i << " at city " << tours[i][j - 1] << endl;
                    exit(EXIT_FAILURE);
                }
                tours[i][j] = nextCity;
                visited[nextCity] = true;
                lengths[i] += distances[tours[i][j - 1]][nextCity];

                cout << "Ant " << i << " city " << j << ": " << nextCity << endl;
            }

            lengths[i] += distances[tours[i][numberOfCities - 1]][tours[i][0]]; // return to start city
            cout << "Ant " << i << " tour length: " << lengths[i] << endl;

            if (lengths[i] < bestTourLength) {
                bestTourLength = lengths[i];
                bestTour = tours[i];
                cout << "New best tour found for ant " << i << ": ";
                for (int j = 0; j < numberOfCities; j++) {
                    cout << bestTour[j] << " ";
                }
                cout << "Length: " << bestTourLength << endl;
            }
        }

        cout << "Iteration " << iteration << " best tour length: " << bestTourLength << endl;

        CUDA_CHECK(cudaMemcpy(d_pheromones, pheromones.data(), pheromonesSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tours, tours.data(), toursSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lengths, lengths.data(), lengthsSize, cudaMemcpyHostToDevice));

        updatePheromones(d_pheromones, d_tours, d_lengths);

        CUDA_CHECK(cudaMemcpy(pheromones.data(), d_pheromones, pheromonesSize, cudaMemcpyDeviceToHost));

        cout << "Pheromone matrix after iteration " << iteration << ":" << endl;
        for (int i = 0; i < numberOfCities; i++) {
            for (int j = 0; j < numberOfCities; j++) {
                cout << pheromones[i][j] << " ";
            }
            cout << endl;
        }
    }

    CUDA_CHECK(cudaFree(d_pheromones));
    CUDA_CHECK(cudaFree(d_tours));
    CUDA_CHECK(cudaFree(d_lengths));
}

int main() {
    initialize();
    findBestTour();

    cout << "Best tour: ";
    for (int i = 0; i < bestTour.size(); i++) {
        cout << bestTour[i] << " ";
    }
    cout << endl;

    cout << "Best tour length: " << bestTourLength << endl;

    return 0;
}
