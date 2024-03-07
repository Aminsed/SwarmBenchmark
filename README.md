# Swarm Algorithms

This repository contains implementations of swarm algorithms, specifically focusing on Particle Swarm Optimization (PSO). The code is written in C++ and includes visualization using Gnuplot.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Swarm algorithms are a class of nature-inspired algorithms that mimic the collective behavior of swarms, such as birds, fish, or insects. These algorithms are used for optimization problems, where the goal is to find the best solution among a set of possible solutions.

Particle Swarm Optimization (PSO) is a popular swarm algorithm that simulates the movement of particles in a search space. Each particle represents a potential solution and adjusts its position based on its own best position and the best position found by the swarm as a whole. Over iterations, the particles converge towards the optimal solution.

## Requirements

To run the code in this repository, you need:

- C++ compiler (supporting C++17 or later)
- Gnuplot (for visualization)

## Installation

1. Clone the repository:

  ```bash
  git clone https://gitlab.com/aminse/swarm-intelligence.git
  ```

2. Navigate to the cloned directory:

  ```bash
  cd swarm-algorithms
  ```

3. Compile the C++ code:

  ```bash
  g++ -std=c++17 pso.cpp -o pso
  ```

## Usage

To run the PSO algorithm and generate the data files for visualization, execute the compiled program:

```bash
./pso
```

The program will run the PSO algorithm and generate the following files:

- `particles_<iteration>.txt`: Contains the positions of all particles at each iteration.
- `best_position.txt`: Contains the global best position found by the algorithm.

The program will also display the global best position, global best value, and execution time in the console.

## Visualization

To visualize the movement of particles over iterations, use the provided Gnuplot script:

```bash
gnuplot -e "ITERATE_MAX=100" animate_particles.plt
```

Make sure to replace `ITERATE_MAX` with the actual number of iterations used in the C++ code.

The Gnuplot script will create an animated visualization of the particles moving towards the optimal solution. The global best position will be highlighted in red.

<img src="https://gitlab.com/aminse/swarm-intelligence/-/blob/main/images/swarm.gif/"/>


## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
