#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

using namespace std;

const double alpha = 1.0; // pheromone importance
const double betaValue = 5.0; // distance priority
const double evaporation = 0.5;
const double Q = 100; // pheromone left on trail per ant
const double antFactor = 0.8; // No. of ants per city
const double randomFactor = 0.01;

int numberOfCities = 20;
int numberOfAnts = (int)(numberOfCities * antFactor);
vector<vector<double>> distances;
vector<vector<double>> pheromones;

vector<int> bestTour;
double bestTourLength = numeric_limits<double>::max();

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
}

void updatePheromones(vector<vector<int>>& tours, vector<double>& lengths) {
    for (int i = 0; i < numberOfCities; i++) {
        for (int j = 0; j < numberOfCities; j++) {
            pheromones[i][j] *= evaporation;
        }
    }

    for (int i = 0; i < numberOfAnts; i++) {
        double contribution = Q / lengths[i];
        for (int j = 0; j < numberOfCities - 1; j++) {
            pheromones[tours[i][j]][tours[i][j + 1]] += contribution;
        }
        pheromones[tours[i][numberOfCities - 1]][tours[i][0]] += contribution; // return to start city
    }
}

double calculateProbability(int from, int to, vector<bool>& visited) {
    double pheromone = pow(pheromones[from][to], alpha);
    double inverseDistance = pow(1.0 / distances[from][to], betaValue);
    if (visited[to]) return 0.0;
    else return pheromone * inverseDistance;
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
    for (int iteration = 0; iteration < 100; iteration++) {
        vector<vector<int>> tours(numberOfAnts, vector<int>(numberOfCities));
        vector<double> lengths(numberOfAnts, 0.0);

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

            lengths[i] += distances[tours[i][numberOfCities - 1]][tours[i][0]]; // return to start city

            if (lengths[i] < bestTourLength) {
                bestTourLength = lengths[i];
                bestTour = tours[i];
            }
        }

        updatePheromones(tours, lengths);
    }
}

int main() {
    initialize();
    findBestTour();
    cout << "Best tour length: " << bestTourLength << endl;
    cout << "Best tour: ";
    for (int i = 0; i < bestTour.size(); i++) {
        cout << bestTour[i] << " ";
    }
    cout << endl;
    return 0;
}
