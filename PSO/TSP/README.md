# Particle Swarm Optimization (PSO) Algorithm

![PSO Particles](https://gitlab.com/aminse/swarm-intelligence/-/raw/main/images/particles.png)

## Introduction

Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the social behavior of bird flocking or fish schooling. It is a metaheuristic that is particularly effective in solving continuous, non-linear, and multi-dimensional optimization problems. This repository contains code that applies PSO to solve the Travelling Salesman Problem, a well-known problem in computer science and operations research.

![TSP](https://i.makeagif.com/media/11-06-2017/6Te1F7.gif)


## Mathematical Formula

The PSO algorithm updates the velocity and position of each particle using the following equations:

```plaintext
v_i(t+1) = w * v_i(t) + c1 * r1 * (p_best_i - x_i(t)) + c2 * r2 * (g_best - x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)
```

Where:
- `v_i(t)`: velocity of particle i at iteration t
- `x_i(t)`: position of particle i at iteration t
- `w`: inertia weight
- `c1`, `c2`: acceleration coefficients
- `r1`, `r2`: random numbers between 0 and 1
- `p_best_i`: personal best position of particle i
- `g_best`: global best position among all particles

### Explanation of the Formula

The PSO algorithm initializes a population of particles, each representing a potential solution. Each particle has a position (`x_i`) and a velocity (`v_i`) in the search space. The position update mechanism is influenced by three components:

1. **Inertia Component**: Maintains the particle's direction.
2. **Cognitive Component**: Drives the particle towards its personal best position.
3. **Social Component**: Encourages movement towards the global best position.

## Pseudo Code

```plaintext
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

1. **Initialization**: Particles are initialized with random positions and velocities.
2. **Fitness Evaluation**: Each particle's fitness is evaluated based on its position.
3. **Best Updates**: Updates are made to personal and global bests based on fitness.
4. **Velocity and Position Update**: Particles' velocities and positions are updated.
5. **Termination**: The process repeats until a stopping criterion is met.

### Reasoning behind the Implementation

The implementation leverages population-based exploration and balances between exploration (searching new areas) and exploitation (refining known good areas) using stochastic components and memory of the best positions.

## Applications of PSO

PSO has been applied to various domains, including:

1. **Travelling Salesman Problem**: Finding the shortest possible route visiting each city once.
2. **Neural Network Training**: Optimizing network weights and biases.
3. **Image and Video Analysis**: Enhancing image segmentation and object detection.
4. **Scheduling and Planning**: Addressing complex scheduling issues.
5. **Antenna Design**: Optimizing antenna configurations for desired performance.
6. **Power System Optimization**: Improving operations in power systems.

These applications demonstrate PSO's versatility and effectiveness in tackling complex optimization challenges.
