# Particle Swarm Optimization (PSO) Algorithm

![Alt Text](https://gitlab.com/aminse/swarm-intelligence/-/raw/main/images/particles.png)

## Introduction
Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the social behavior of bird flocking or fish schooling. It is a metaheuristic algorithm that can be used to solve a wide range of optimization problems. PSO is particularly effective in solving continuous, non-linear, and multi-dimensional optimization problems. In this repository, the code will attempt to solve the Rastrigin function, a non-convex function used as a performance test problem for optimization algorithms.

![Alt Text](https://upload.wikimedia.org/wikipedia/commons/8/8b/Rastrigin_function.png)

## Mathematical Formula
The PSO algorithm is based on the following mathematical formula:

```
v_i(t+1) = w * v_i(t) + c1 * r1 * (p_best_i - x_i(t)) + c2 * r2 * (g_best - x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)
```

Visualization of PSO:

![Alt Text](https://gitlab.com/aminse/swarm-intelligence/-/raw/main/images/swarm.gif)

Where:
- `v_i(t)`: velocity of particle i at iteration t
- `x_i(t)`: position of particle i at iteration t
- `w`: inertia weight
- `c1`, `c2`: acceleration coefficients
- `r1`, `r2`: random numbers between 0 and 1
- `p_best_i`: personal best position of particle i
- `g_best`: global best position among all particles

### Explanation of the Formula
The PSO algorithm works by initializing a population of particles, each representing a potential solution to the optimization problem. Each particle has a position (`x_i`) and a velocity (`v_i`) in the search space.

The position of a particle represents a candidate solution, while the velocity determines the direction and magnitude of the particle's movement in the search space.

The velocity update equation consists of three components:
1. **Inertia Component**: `w * v_i(t)` represents the particle's tendency to continue moving in its current direction. The inertia weight `w` controls the influence of the previous velocity on the current velocity.

2. **Cognitive Component**: `c1 * r1 * (p_best_i - x_i(t))` represents the particle's attraction towards its personal best position (`p_best_i`). It encourages the particle to move towards the best solution it has found so far. The acceleration coefficient `c1` determines the influence of the personal best, and `r1` is a random number that introduces stochasticity.

3. **Social Component**: `c2 * r2 * (g_best - x_i(t))` represents the particle's attraction towards the global best position (`g_best`) found by the entire swarm. It encourages the particle to move towards the best solution found by any particle in the swarm. The acceleration coefficient `c2` determines the influence of the global best, and `r2` is a random number.

The updated velocity is then added to the current position of the particle to obtain its new position in the search space.

## Pseudo Code
```
Initialize a population of particles with random positions and velocities
while termination condition is not met:
    for each particle:
        Calculate fitness value
        if fitness value is better than the personal best (p_best):
            Update personal best (p_best)
        if fitness value is better than the global best (g_best):
            Update global best (g_best)
    for each particle:
        Update velocity using the velocity update equation
        Update position using the position update equation
```

### Explanation of the Pseudo Code
1. **Initialization**: The algorithm starts by initializing a population of particles with random positions and velocities in the search space. Each particle represents a potential solution to the optimization problem.

2. **Fitness Evaluation**: For each particle, the fitness value (objective function) is calculated based on its current position. The fitness value measures the quality of the solution represented by the particle.

3. **Personal Best Update**: If the current fitness value of a particle is better than its personal best (p_best) fitness value, the personal best position is updated to the current position.

4. **Global Best Update**: If the current fitness value of a particle is better than the global best (g_best) fitness value found so far by any particle in the swarm, the global best position is updated to the current position of that particle.

5. **Velocity Update**: For each particle, the velocity is updated using the velocity update equation mentioned earlier. The updated velocity takes into account the particle's current velocity, its attraction towards its personal best, and its attraction towards the global best.

6. **Position Update**: The position of each particle is updated by adding the updated velocity to its current position. This moves the particle to a new position in the search space.

7. **Termination**: The algorithm repeats steps 2-6 until a termination condition is met. The termination condition can be a maximum number of iterations, a desired fitness value, or any other problem-specific criterion.

### Reasoning behind the Implementation
The PSO algorithm is implemented in this way to efficiently explore the search space and find optimal solutions. The key aspects of the implementation are:

- **Population-based**: PSO uses a population of particles to explore the search space simultaneously. This allows for a diverse set of solutions to be considered and helps avoid getting stuck in local optima.

- **Personal and Global Best**: The personal best (p_best) and global best (g_best) positions guide the particles towards promising regions of the search space. The personal best encourages each particle to explore around its own best solution, while the global best encourages particles to move towards the best solution found by the entire swarm.

- **Velocity Update**: The velocity update equation balances exploration and exploitation. The inertia component allows particles to maintain their current direction, the cognitive component encourages particles to explore their personal best regions, and the social component encourages particles to move towards the global best region.

- **Stochasticity**: The random numbers (`r1` and `r2`) introduce stochasticity into the velocity update equation. This helps in exploring different regions of the search space and prevents premature convergence to suboptimal solutions.

## Applications of PSO
The Particle Swarm Optimization algorithm has been successfully applied to a wide range of optimization problems across various domains. Some prominent examples include:

1. **Function Optimization**: PSO has been used to optimize complex mathematical functions, including multi-modal and high-dimensional functions. Specifically, this repository focuses on solving the Rastrigin function, demonstrating PSO's capability in handling complex, non-linear optimization challenges.

2. **Neural Network Training**: PSO has been employed to optimize the weights and biases of artificial neural networks, improving their training process and generalization performance.

3. **Image and Video Analysis**: PSO has been applied to tasks such as image segmentation, object detection, and video tracking, where it helps in finding optimal parameters for image processing algorithms.

4. **Scheduling and Planning**: PSO has been used to solve scheduling and planning problems, such as job shop scheduling, vehicle routing, and resource allocation.

5. **Antenna Design**: PSO has been utilized to optimize the design of antennas, including the placement and configuration of antenna elements to achieve desired radiation patterns and performance characteristics.

6. **Power System Optimization**: PSO has been applied to optimize power system operations, including economic dispatch, load flow analysis, and optimal power flow problems.

These are just a few examples of the diverse range of applications where the Particle Swarm Optimization algorithm has been successfully employed. PSO's simplicity, flexibility, and ability to handle complex optimization problems make it a popular choice across various fields.