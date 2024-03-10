#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <execution>
#include <omp.h>

using namespace std;

class AntColonyOptimization {
private:
    const double alpha = 1.0;
    const double betaValue = 5.0;
    const double evaporation = 0.5;
    const double Q = 100;
    const double antFactor = 0.8;
    const double randomFactor = 0.01;

    int numberOfCities;
    int numberOfAnts;
    vector<vector<double>> distances;
    vector<vector<double>> pheromones;

    vector<int> bestTour;
    double bestTourLength = numeric_limits<double>::max();

    mt19937 randomEngine{random_device{}()};

public:
    AntColonyOptimization(int numCities) : numberOfCities(numCities), numberOfAnts(static_cast<int>(numCities * antFactor)) {
        distances.resize(numberOfCities, vector<double>(numberOfCities));
        pheromones.resize(numberOfCities, vector<double>(numberOfCities, 1.0));

        uniform_int_distribution<int> distanceDistribution(1, 100);
        for (int i = 0; i < numberOfCities; i++) {
            for (int j = i + 1; j < numberOfCities; j++) {
                double dist = distanceDistribution(randomEngine);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
    }

    void updatePheromones(const vector<vector<int>>& tours, const vector<double>& lengths) {
        for_each(execution::par, pheromones.begin(), pheromones.end(), [this](auto& row) {
            for_each(execution::par, row.begin(), row.end(), [this](auto& pheromone) {
                pheromone *= evaporation;
            });
        });

        for (int i = 0; i < numberOfAnts; i++) {
            double contribution = Q / lengths[i];
            for (int j = 0; j < numberOfCities - 1; j++) {
                pheromones[tours[i][j]][tours[i][j + 1]] += contribution;
            }
            pheromones[tours[i][numberOfCities - 1]][tours[i][0]] += contribution;
        }
    }

    double calculateProbability(int from, int to, const vector<bool>& visited) {
        if (visited[to]) return 0.0;
        double pheromone = pow(pheromones[from][to], alpha);
        double inverseDistance = pow(1.0 / distances[from][to], betaValue);
        return pheromone * inverseDistance;
    }

    int selectNextCity(int currentCity, vector<bool>& visited) {
        vector<double> probabilities(numberOfCities);
        double sumProbabilities = 0.0;

        for (int i = 0; i < numberOfCities; i++) {
            if (!visited[i]) {
                probabilities[i] = calculateProbability(currentCity, i, visited);
                sumProbabilities += probabilities[i];
            }
        }

        if (sumProbabilities == 0) return -1;

        uniform_real_distribution<double> distribution(0.0, sumProbabilities);
        double random = distribution(randomEngine);
        double cumulativeProbability = 0.0;
        for (int i = 0; i < numberOfCities; i++) {
            if (!visited[i]) {
                cumulativeProbability += probabilities[i];
                if (random <= cumulativeProbability) {
                    return i;
                }
            }
        }

        return -1;
    }

    void findBestTour() {
        for (int iteration = 0; iteration < 100; iteration++) {
            vector<vector<int>> tours(numberOfAnts, vector<int>(numberOfCities));
            vector<double> lengths(numberOfAnts, 0.0);

            #pragma omp parallel for
            for (int i = 0; i < numberOfAnts; i++) {
                vector<bool> visited(numberOfCities, false);
                tours[i][0] = rand() % numberOfCities;
                visited[tours[i][0]] = true;

                for (int j = 1; j < numberOfCities; j++) {
                    int nextCity = selectNextCity(tours[i][j - 1], visited);
                    tours[i][j] = nextCity;
                    visited[nextCity] = true;
                    lengths[i] += distances[tours[i][j - 1]][nextCity];
                }

                lengths[i] += distances[tours[i][numberOfCities - 1]][tours[i][0]];

                #pragma omp critical
                {
                    if (lengths[i] < bestTourLength) {
                        bestTourLength = lengths[i];
                        bestTour = tours[i];
                    }
                }
            }

            updatePheromones(tours, lengths);
        }
    }

    void printResult() {
        cout << "Best tour length: " << bestTourLength << endl;
        cout << "Best tour: ";
        for (int city : bestTour) {
            cout << city << " ";
        }
        cout << endl;
    }
};

int main() {
    int numberOfCities = 20;
    AntColonyOptimization aco(numberOfCities);
    aco.findBestTour();
    aco.printResult();
    return 0;
}
