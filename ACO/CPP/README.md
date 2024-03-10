# Ant Colony Optimization for the Traveling Salesman Problem

This document outlines the implementation of an Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesman Problem (TSP). The algorithm simulates the behavior of ants to find the shortest possible tour that visits a set of cities.

## Table of Contents

- [Section 1: Initialization](#section-1-initialization)
- [Section 2: Pheromone Update](#section-2-pheromone-update)
- [Section 3: Probability Calculation](#section-3-probability-calculation)
- [Section 4: Next City Selection](#section-4-next-city-selection)
- [Section 5: Best Tour Update](#section-5-best-tour-update)
- [Section 6: Main Function](#section-6-main-function)

## Section 1: Initialization

### Pseudo Code

```plaintext
function initialize()
    resize distances and pheromones vectors
    set initial pheromone levels
    generate random distances between cities
```

### Explanation

The `initialize()` function sets up the initial state of the problem. It resizes the `distances` and `pheromones` vectors based on the number of cities. The `distances` vector stores the distances between each pair of cities, while the `pheromones` vector stores the pheromone levels on the paths between cities.

The initial pheromone levels are set to 1 for all paths. This ensures that all paths have an equal probability of being chosen by the ants at the beginning.

Random distances between cities are generated using the `rand()` function and stored in the `distances` vector. The distances are symmetric, meaning that the distance from city i to city j is the same as the distance from city j to city i.

### Reasoning

Initializing the problem with random distances allows for the simulation of different TSP instances. By starting with equal pheromone levels on all paths, the ants have no initial bias towards any particular path. This allows the algorithm to explore different solutions and gradually converge towards the optimal path based on the pheromone updates.

## Section 2: Pheromone Update

### Pseudo Code

```plaintext
function updatePheromones(tours, lengths)
    evaporate pheromone on all paths
    for each ant's tour
        deposit pheromone based on tour length
```

### Explanation

The `updatePheromones()` function is responsible for updating the pheromone levels on the paths based on the quality of the tours found by the ants.

First, the pheromone on all paths is evaporated by multiplying the current pheromone levels by the `evaporation` rate. This simulates the natural decay of pheromones over time and allows the algorithm to forget older solutions.

Next, for each ant's tour, pheromone is deposited on the paths that the ant traversed. The amount of pheromone deposited is inversely proportional to the length of the tour. Shorter tours deposit more pheromone, while longer tours deposit less pheromone. The pheromone deposit is calculated as `Q / tourLength`, where `Q` is a constant that determines the overall pheromone deposit amount.

### Reasoning

The pheromone update process is a crucial part of the ACO algorithm. By evaporating pheromone on all paths, the algorithm prevents the pheromone levels from accumulating indefinitely and allows for the exploration of new solutions. Depositing pheromone based on the quality of the tours reinforces the paths that lead to good solutions. Ants are more likely to follow paths with higher pheromone levels, which guides the search towards promising regions of the solution space.

## Section 3: Probability Calculation

### Pseudo Code

```plaintext
function calculateProbability(from, to, visited)
    calculate probability based on pheromone and distance
    if destination city is visited, probability = 0
    return probability
```

### Explanation

The `calculateProbability()` function calculates the probability of an ant moving from one city to another based on the pheromone level and the distance between the cities.

The probability is calculated using the formula: `pheromone^alpha * (1/distance)^beta`. The `alpha` parameter determines the importance of the pheromone level, while the `beta` parameter determines the importance of the distance. Higher values of `alpha` give more weight to the pheromone level, while higher values of `beta` give more weight to the distance.

If the destination city has already been visited by the ant, the probability is set to 0 to prevent the ant from revisiting the same city.

### Reasoning

The probability calculation incorporates both the pheromone level and the distance to guide the ants' decision-making process. The pheromone level represents the collective knowledge gained by previous ants, while the distance represents the cost of traveling between cities. By combining these two factors, the ants are more likely to choose paths that have a good balance between pheromone intensity and distance.

Setting the probability to 0 for visited cities ensures that each ant constructs a valid tour that visits each city exactly once.

## Section 4: Next City Selection

### Pseudo Code

```plaintext
function selectNextCity(currentCity, visited)
    calculate probabilities for unvisited cities
    normalize probabilities
    use roulette wheel selection to choose next city
    return selected city
```

### Explanation

The `selectNextCity()` function is responsible for selecting the next city for an ant based on the calculated probabilities.

First, the probabilities for all unvisited cities are calculated using the `calculateProbability()` function. These probabilities are stored in a vector.

Next, the probabilities are normalized by dividing each probability by the sum of all probabilities. This ensures that the probabilities sum up to 1.

Finally, roulette wheel selection is used to choose the next city probabilistically. The roulette wheel selection works by generating a random number between 0 and 1 and comparing it with the cumulative probabilities of the cities. The city whose cumulative probability range contains the random number is selected as the next city.

### Reasoning

The next city selection process uses a probabilistic approach to balance exploration and exploitation. By calculating probabilities based on pheromone levels and distances, the ants are more likely to choose promising paths. However, the randomness introduced by roulette wheel selection allows for the exploration of new solutions and prevents the algorithm from getting stuck in local optima.

Normalizing the probabilities ensures that the selection process is fair and unbiased, giving each city a chance to be selected proportional to its probability.

## Section 5: Best Tour Update

### Pseudo Code

```plaintext
function findBestTour()
    for iteration = 1 to maxIterations
        for each ant
            construct tour
            calculate tour length
            update best tour if shorter
        update pheromone levels
```

### Explanation

The `findBestTour()` function implements the main loop of the ACO algorithm. It repeats the process of constructing tours, updating the best tour, and updating pheromone levels for a specified number of iterations.

In each iteration, each ant constructs a tour by starting at a random city and selecting the next city using the `selectNextCity()` function until all cities are visited. The tour length is calculated by summing up the distances between the visited cities.

If a shorter tour is found by an ant, the best tour and its length are updated accordingly.

After all ants have constructed their tours, the pheromone levels are updated using the `updatePheromones()` function.

### Reasoning

The iterative process of constructing tours and updating pheromone levels allows the ACO algorithm to progressively improve the solution quality. By repeating this process for a sufficient number of iterations, the algorithm converges towards the optimal or near-optimal solution.

Updating the best tour whenevera shorter tour is found ensures that the algorithm keeps track of the best solution encountered throughout the iterations.

## Section 6: Main Function

### Pseudo Code

```plaintext
function main()
    initialize problem
    find best tour
    print best tour length and path
```

### Explanation

The `main()` function serves as the entry point of the program. It orchestrates the overall flow of the ACO algorithm.

First, the problem is initialized by calling the `initialize()` function, which sets up the distances and pheromone levels.

Next, the `findBestTour()` function is called to execute the main loop of the ACO algorithm and find the best tour.

Finally, the best tour length and the corresponding tour path are printed as the output of the program.

### Reasoning

The main function provides a high-level overview of the program's execution. It encapsulates the initialization, solution finding, and output steps, making the code more modular and easier to understand.

By separating the problem initialization, algorithm execution, and output, the code becomes more readable and maintainable.