# Particle Swarm Optimization (PSO) with OpenMP

This repository contains an implementation of the Particle Swarm Optimization (PSO) algorithm using C++ and OpenMP for parallelization. The PSO algorithm is used to find the global minimum of the Rastrigin function in a 2D search space.

## Requirements

- C++ compiler with C++17 support (e.g., g++, clang++)
- OpenMP library

## Compilation

To compile the code, navigate to the directory containing the `pso.cpp` file and run the following command:

```bash
g++ -std=c++17 -O3 -fopenmp pso.cpp -o pso
```

This command compiles the `pso.cpp` file using g++ with C++17 standard, a high level of optimization (`-O3`), and OpenMP support (`-fopenmp`). The resulting executable will be named `pso`.

If you are using a different compiler, make sure to use the appropriate compiler command and flags.

## Running the Program

To run the program, use the following command:

```bash
./pso
```

The program will execute, and you will see the output displaying the global best position, global best value, and execution time.

## Customization

You can customize the following parameters in the `pso.cpp` file:

- `NUM_PARTICLES`: The number of particles in the swarm.
- `MAX_ITERATIONS`: The maximum number of iterations for the PSO algorithm.
- `SEARCH_SPACE_MIN`: The minimum value of the search space.
- `SEARCH_SPACE_MAX`: The maximum value of the search space.
- `INERTIA_WEIGHT`: The inertia weight parameter.
- `COGNITIVE_WEIGHT`: The cognitive weight parameter.
- `SOCIAL_WEIGHT`: The social weight parameter.

Feel free to experiment with different values of these parameters to observe their impact on the optimization process.

## Parallelization

The code utilizes OpenMP directives to parallelize the loops for initializing particles and updating particle positions and velocities. The `#pragma omp parallel for` directive is used to distribute the loop iterations among multiple threads.

The `#pragma omp critical` directive is used to ensure thread-safe access when updating the global best position and value.

## Optimization

The code is compiled with a high level of optimization (`-O3`) to improve performance. You can adjust the optimization level by changing the number after `-O` (e.g., `-O2` for a lower level of optimization).

## License

This code is released under the [MIT License](https://opensource.org/licenses/MIT).
