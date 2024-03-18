# Firefly Algorithm

![Alt Text](https://gitlab.com/aminse/swarm-intelligence/-/raw/main/images/fire_fly.png)

## Introduction
Firefly Algorithm (FA) is a population-based optimization algorithm inspired by the flashing behavior of fireflies. It is a metaheuristic algorithm that can be used to solve a wide range of optimization problems. FA is particularly effective in solving continuous, non-linear, and multi-dimensional optimization problems. In this repository, the code will attempt to solve the Rosenbrock function, a non-convex function used as a performance test problem for optimization algorithms.

![Alt Text](https://www.researchgate.net/publication/297617254/figure/fig1/AS:600887030472708@1520274040915/The-Rosenbrock-function-a-classic-test-problem-in-optimization.png)

## Mathematical Formula
The Firefly Algorithm is based on the following mathematical formulas:

```
Attractiveness: β(r) = β₀ * exp(-γ * r²)
Distance: r_ij = ||x_i - x_j||
Movement: x_i = x_i + β(r_ij) * (x_j - x_i) + α * ε_i
```

Where:
- `β(r)`: attractiveness of a firefly at a distance r
- `β₀`: attractiveness at r = 0
- `γ`: light absorption coefficient
- `r_ij`: Euclidean distance between fireflies i and j
- `x_i`: position of firefly i
- `α`: randomization parameter
- `ε_i`: random vector drawn from a Gaussian or uniform distribution

### Explanation of the Formula
The Firefly Algorithm works by considering a population of fireflies, each representing a potential solution to the optimization problem. The attractiveness of a firefly determines its ability to attract other fireflies, and it decreases with distance.

The distance between two fireflies is calculated using the Euclidean distance formula, which measures the straight-line distance between their positions in the search space.

The movement of a firefly is influenced by three components:
1. **Attractiveness**: A firefly is attracted to other fireflies based on their attractiveness. The magnitude of attraction is determined by the attractiveness function `β(r)`, which decreases exponentially with the square of the distance between fireflies.

2. **Randomization**: The movement of a firefly is also influenced by a randomization parameter `α`, which controls the step size of the random movement. This allows fireflies to explore the search space and escape local optima.

3. **Current Position**: The new position of a firefly is determined by its current position, the attraction towards other fireflies, and the random movement.

## Pseudo Code
```
Initialize a population of fireflies with random positions
while termination condition is not met:
    for each firefly i:
        for each firefly j:
            if attractiveness of j > attractiveness of i:
                Move firefly i towards firefly j
        Update attractiveness of firefly i
    Rank fireflies based on their attractiveness
    Update the global best solution
```

### Explanation of the Pseudo Code
1. **Initialization**: The algorithm starts by initializing a population of fireflies with random positions in the search space. Each firefly represents a potential solution to the optimization problem.

2. **Attractiveness Comparison**: For each firefly i, the algorithm compares its attractiveness with every other firefly j. If the attractiveness of firefly j is greater than the attractiveness of firefly i, firefly i is moved towards firefly j based on the movement formula.

3. **Attractiveness Update**: After the movement, the attractiveness of each firefly is updated based on the attractiveness function `β(r)`.

4. **Ranking**: The fireflies are ranked based on their updated attractiveness values. The firefly with the highest attractiveness represents the best solution found so far.

5. **Global Best Update**: The global best solution is updated if a firefly with higher attractiveness is found.

6. **Termination**: The algorithm repeats steps 2-5 until a termination condition is met. The termination condition can be a maximum number of iterations, a desired fitness value, or any other problem-specific criterion.

### Reasoning behind the Implementation
The Firefly Algorithm is implemented in this way to efficiently explore the search space and find optimal solutions. The key aspects of the implementation are:

- **Attractiveness-based Movement**: The movement of fireflies is guided by their attractiveness. Fireflies are attracted to brighter fireflies, which leads to the exploration of promising regions in the search space.

- **Distance-based Attractiveness**: The attractiveness of a firefly decreases with distance, ensuring that the influence of distant fireflies is limited. This allows the algorithm to focus on local search while still considering the global perspective.

- **Randomization**: The randomization parameter introduces stochasticity into the movement of fireflies. It helps in escaping local optima and exploring different regions of the search space.

- **Ranking and Global Best**: The ranking of fireflies based on their attractiveness helps identify the best solutions found so far. The global best solution is updated whenever a better solution is discovered.

## Applications of Firefly Algorithm
The Firefly Algorithm has been applied to various optimization problems across different domains. Some notable applications include:

1. **Continuous Optimization**: The Firefly Algorithm has been used to solve continuous optimization problems, such as function optimization, parameter estimation, and curve fitting.

2. **Combinatorial Optimization**: The algorithm has been adapted to solve combinatorial optimization problems, including scheduling, routing, and resource allocation.

3. **Image Processing**: The Firefly Algorithm has been employed in image processing tasks, such as image compression, image segmentation, and feature selection.

4. **Wireless Sensor Networks**: The algorithm has been applied to optimize the deployment and routing of nodes in wireless sensor networks.

5. **Engineering Design**: The Firefly Algorithm has been used to optimize the design of various engineering systems, such as structural design, mechanical design, and electrical circuit design.

These are just a few examples of the diverse range of applications where the Firefly Algorithm has been successfully applied. The algorithm's simplicity, flexibility, and ability to handle complex optimization problems make it a popular choice in various domains.
