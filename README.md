# Swarm Intelligence Algorithms

Welcome to the Swarm Intelligence Algorithms Repository! This repository contains implementations of various swarm intelligence algorithms in C++, OpenMP, CUDA, and Thrust. The goal of this repository is to provide a comprehensive collection of swarm intelligence algorithms for educational and research purposes.

## Algorithms Included

1. [Particle Swarm Optimization (PSO)](./PSO/None-TSP/)
2. [Firefly Algorithm (FA)](./FA)
3. [Moth-Flame Optimization (MFO) Algorithm](./MFO)
4. [Grey Wolf Optimizer (GWO)](./GWO)
5. [Whale Optimization Algorithm (WOA)](./WOA)
6. [Artificial Bee Colony (ABC) Algorithm](./ABC/None-TSP/)
7. [Dragonfly Algorithm (DA)](./DA)
8. [Salp Swarm Algorithm (SSA)](./SSA/)
9. [Grasshopper Optimization Algorithm (GOA)](./GOA)
10. [Ant Colony Optimization (ACO)](./ACO/None-TSP/)

## Repository Structure

Each algorithm has its own dedicated folder, named after the algorithm's abbreviation (e.g., PSO, ACO, FA). Within each algorithm folder, you will find four subfolders:

- `CPP`: Contains the pure C++ implementation of the algorithm.
- `OMP`: Contains the OpenMP implementation of the algorithm for parallel processing.
- `THRUST`: Contains the THRUST implementation of the algorithm for GPU acceleration.
- `CUDA`: Contains the CUDA implementation of the algorithm for GPU acceleration.

Inside each subfolder, you will find the respective code files and a README file with instructions on how to compile and run the code.

## Getting Started

To compile and run CPP and OMP codes, make sure to have GCC installed. For CUDA and THRUST codes, make sure to have NVCC installed. Additionally, ensure CMake is installed for all. Simply navigate to the desired algorithm's folder and run `make` to generate the executable file. After you're done running the executable, you can run `make clean` to remove all of the generated files.

## Benchmarking Details

The benchmarking results presented in this repository were obtained under controlled conditions to ensure comparability across different implementations. Below are the details of the testing environment and parameters:

- **Function Used for Testing**: Rastrigin Function. Lower `Execution Time` and `Result` means better performance.
- **Parameters**: All algorithms were tested using identical parameters to ensure fair comparison.
- **Hardware Specifications**:
  - **CPU**: Intel i9-12900K
  - **GPU**: NVIDIA RTX A6000 (Ampere Architecture)

These specifications were maintained consistently to evaluate the performance of each algorithm implementation accurately.

## Contributing

Contributions to this repository are welcome! If you have an implementation of a swarm intelligence algorithm that is not currently included, or if you want to improve an existing implementation, please submit a pull request. Make sure to follow the repository's structure and provide clear instructions on how to compile and run your code.

## Further Reading

For a detailed benchmarking study using the findings of this repository, you can read the article [Swarm Intelligence Showdown: A Benchmarking Study](https://medium.com/@amin32846/swarm-intelligence-showdown-a-benchmarking-study-a94cc2ca598c) on Medium.

## License

This repository is licensed under the [MIT License](LICENSE). Feel free to use the code for educational and research purposes. If you use any of the implementations in your work, please consider citing this repository.
