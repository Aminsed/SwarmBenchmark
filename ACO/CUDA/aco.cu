// ... (previous code remains the same)

__global__ void antKernel(double* d_distances, double* d_pheromones, int* d_tours, double* d_lengths, curandState* state) {
    int antId = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ bool visited[];
    double* probabilities = (double*)&visited[d_numberOfCities];

    if (antId < d_numberOfAnts) {
        for (int i = 0; i < d_numberOfCities; i++) {
            visited[i] = false;
        }

        int startCity = curand(&state[antId]) % d_numberOfCities;
        visited[startCity] = true;
        d_tours[antId * d_numberOfCities] = startCity;

        for (int i = 1; i < d_numberOfCities; i++) {
            int currentCity = d_tours[antId * d_numberOfCities + i - 1];
            double sumProbabilities = 0.0;

            for (int j = 0; j < d_numberOfCities; j++) {
                if (!visited[j]) {
                    double pheromone = pow(d_pheromones[currentCity * d_numberOfCities + j], d_alpha);
                    double distance = d_distances[currentCity * d_numberOfCities + j];
                    double probability = pheromone / distance;
                    probabilities[j] = probability;
                    sumProbabilities += probability;
                }
            }

            double random = curand_uniform(&state[antId]);
            double cumulativeProbability = 0.0;
            int nextCity = -1;

            for (int j = 0; j < d_numberOfCities; j++) {
                if (!visited[j]) {
                    probabilities[j] /= sumProbabilities;
                    cumulativeProbability += probabilities[j];
                    if (random <= cumulativeProbability) {
                        nextCity = j;
                        break;
                    }
                }
            }

            if (nextCity == -1) {
                printf("Error: No unvisited city found for ant %d at iteration %d\n", antId, i);
            }

            d_tours[antId * d_numberOfCities + i] = nextCity;
            visited[nextCity] = true;
            d_lengths[antId] += d_distances[currentCity * d_numberOfCities + nextCity];
        }

        d_lengths[antId] += d_distances[d_tours[antId * d_numberOfCities + d_numberOfCities - 1] * d_numberOfCities + startCity];

        printf("Ant %d tour length: %f\n", antId, d_lengths[antId]);
    }
}

__global__ void updatePheromonesKernel(double* d_pheromones, int* d_tours, double* d_lengths) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < d_numberOfCities && y < d_numberOfCities) {
        d_pheromones[x * d_numberOfCities + y] *= evaporation;

        for (int i = 0; i < d_numberOfAnts; i++) {
            double contribution = Q / d_lengths[i];
            for (int j = 0; j < d_numberOfCities - 1; j++) {
                int city1 = d_tours[i * d_numberOfCities + j];
                int city2 = d_tours[i * d_numberOfCities + j + 1];
                if ((city1 == x && city2 == y) || (city1 == y && city2 == x)) {
                    d_pheromones[x * d_numberOfCities + y] += contribution;
                }
            }
            int lastCity = d_tours[i * d_numberOfCities + d_numberOfCities - 1];
            int firstCity = d_tours[i * d_numberOfCities];
            if ((lastCity == x && firstCity == y) || (lastCity == y && firstCity == x)) {
                d_pheromones[x * d_numberOfCities + y] += contribution;
            }
        }
    }
}

void findBestTour(double* d_distances, double* d_pheromones, int* d_tours, double* d_lengths, curandState* state) {
    for (int iteration = 0; iteration < 100; iteration++) {
        initializeLengthsKernel<<<(numberOfAnts + 255) / 256, 256>>>(d_lengths);
        cudaDeviceSynchronize();

        antKernel<<<(numberOfAnts + 255) / 256, 256, (numberOfCities * sizeof(bool)) + (numberOfCities * sizeof(double))>>>(d_distances, d_pheromones, d_tours, d_lengths, state);
        cudaDeviceSynchronize();

        double h_lengths[numberOfAnts];
        cudaMemcpy(h_lengths, d_lengths, numberOfAnts * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numberOfAnts; i++) {
            printf("Iteration %d, Ant %d tour length: %f\n", iteration, i, h_lengths[i]);

            if (h_lengths[i] < bestTourLength) {
                bestTourLength = h_lengths[i];
                int h_tour[numberOfCities];
                cudaMemcpy(h_tour, d_tours + i * numberOfCities, numberOfCities * sizeof(int), cudaMemcpyDeviceToHost);
                bestTour.assign(h_tour, h_tour + numberOfCities);

                printf("New best tour found at iteration %d, Ant %d, Length: %f\n", iteration, i, bestTourLength);
            }
        }

        dim3 blockSize(32, 32);
        dim3 gridSize((numberOfCities + blockSize.x - 1) / blockSize.x, (numberOfCities + blockSize.y - 1) / blockSize.y);
        updatePheromonesKernel<<<gridSize, blockSize>>>(d_pheromones, d_tours, d_lengths);
        cudaDeviceSynchronize();
    }
}


int main() {
    numberOfAnts = (int)(numberOfCities * antFactor);

    double h_distances[numberOfCities * numberOfCities];
    double h_pheromones[numberOfCities * numberOfCities];

    for (int i = 0; i < numberOfCities; i++) {
        for (int j = i + 1; j < numberOfCities; j++) {
            double dist = rand() % 100 + 1;
            h_distances[i * numberOfCities + j] = dist;
            h_distances[j * numberOfCities + i] = dist;
        }
        h_distances[i * numberOfCities + i] = 0; // Set distance to itself as 0
    }

    fill(h_pheromones, h_pheromones + numberOfCities * numberOfCities, 1.0);

    double* d_distances;
    double* d_pheromones;
    int* d_tours;
    double* d_lengths;
    curandState* d_state;

    cudaMalloc((void**)&d_distances, numberOfCities * numberOfCities * sizeof(double));
    cudaMalloc((void**)&d_pheromones, numberOfCities * numberOfCities * sizeof(double));
    cudaMalloc((void**)&d_tours, numberOfAnts * numberOfCities * sizeof(int));
    cudaMalloc((void**)&d_lengths, numberOfAnts * sizeof(double));
    cudaMalloc((void**)&d_state, numberOfAnts * sizeof(curandState));

    cudaMemcpy(d_distances, h_distances, numberOfCities * numberOfCities * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pheromones, h_pheromones, numberOfCities * numberOfCities * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_numberOfCities, &numberOfCities, sizeof(int));
    cudaMemcpyToSymbol(d_numberOfAnts, &numberOfAnts, sizeof(int));
    cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(double));
    cudaMemcpyToSymbol(d_betaValue, &betaValue, sizeof(double));

    setupKernel<<<(numberOfAnts + 255) / 256, 256>>>(d_state);

    findBestTour(d_distances, d_pheromones, d_tours, d_lengths, d_state);

    cout << "Best tour length: " << bestTourLength << endl;
    cout << "Best tour: ";
    for (int i = 0; i < bestTour.size(); i++) {
        cout << bestTour[i] << " ";
    }
    cout << endl;

    cudaFree(d_distances);
    cudaFree(d_pheromones);
    cudaFree(d_tours);
    cudaFree(d_lengths);
    cudaFree(d_state);

    return 0;
}
