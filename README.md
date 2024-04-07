# Swarm Intelligence Algorithms

Welcome to the Swarm Intelligence Algorithms Repository! This repository contains implementations of various swarm intelligence algorithms in C++, OpenMP, CUDA, and Thrust. The goal of this repository is to provide a comprehensive collection of swarm intelligence algorithms for educational and research purposes.

## Algorithms Included

1. Particle Swarm Optimization (PSO) ✅
   - CPP ✅  
     - Execution Time: 80.00 ms
   - OMP ✅  
     - Execution Time: 3560.00 ms
   - THRUST ✅  
     - Execution Time: 11.00 ms
   - CUDA ✅  
     - Execution Time: 11.00 ms
2. Firefly Algorithm (FA) ✅
   - CPP ✅
     - Execution Time: 26885.00 ms
   - OMP ✅
     - Execution Time: 35565.00 ms
   - THRUST ✅
     - Execution Time: 43861.00 ms
   - CUDA ✅
     - Execution Time: 43846.00 ms
3. Moth-Flame Optimization (MFO) Algorithm ✅
   - CPP ✅
   - OMP ✅
   - THRUST ✅
   - CUDA ✅
4. Grey Wolf Optimizer (GWO) ✅
   - CPP ✅
   - OMP ✅
   - THRUST ✅
   - CUDA ✅
5. Whale Optimization Algorithm (WOA) ✅
   - CPP ✅
   - OMP ✅
   - CUDA ✅
   - THRUST ✅
6. Artificial Bee Colony (ABC) Algorithm ✅
   - CPP ✅
   - OMP ✅
   - THRUST ✅
   - CUDA ✅
7. Dragonfly Algorithm (DA)
   - CPP
   - OMP
   - THRUST
   - CUDA
8. Salp Swarm Algorithm (SSA)
   - CPP
   - OMP
   - THRUST
   - CUDA
9. Grasshopper Optimization Algorithm (GOA)
   - CPP
   - OMP
   - THRUST
   - CUDA
10. Ant Colony Optimization (ACO)
    - CPP ✅
    - OMP ✅
    - THRUST
    - CUDA

## Repository Structure

Each algorithm has its own dedicated folder, named after the algorithm's abbreviation (e.g., PSO, ACO, FA). Within each algorithm folder, you will find four subfolders:

- `CPP`: Contains the pure C++ implementation of the algorithm.
- `OMP`: Contains the OpenMP implementation of the algorithm for parallel processing.
- `THRUST`: Contains the THRUST implementation of the algorithm for GPU acceleration.
- `CUDA`: Contains the CUDA implementation of the algorithm for GPU acceleration.

Inside each subfolder, you will find the respective code files and a README file with instructions on how to compile and run the code.

## Getting Started

To compile and run CPP and OMP codes, make sure to have GCC installed. For CUDA and THRUST codes, make sure to have NVCC installed. Additionally, ensure CMake is installed for all. Simply navigate to the desired algorithm's folder and run `make` to generate the executable file. After you're done running the executable, you can run `make clean` to remove all of the generated files.

## Contributing

Contributions to this repository are welcome! If you have an implementation of a swarm intelligence algorithm that is not currently included, or if you want to improve an existing implementation, please submit a pull request. Make sure to follow the repository's structure and provide clear instructions on how to compile and run your code.

## License

This repository is licensed under the [MIT License](LICENSE). Feel free to use the code for educational and research purposes. If you use any of the implementations in your work, please consider citing this repository.