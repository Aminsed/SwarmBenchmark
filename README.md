# Swarm Intelligence Algorithms

Welcome to the Swarm Intelligence Algorithms Repository! This repository contains implementations of various swarm intelligence algorithms in C++, OpenMP, CUDA, and Thrust. The goal of this repository is to provide a comprehensive collection of swarm intelligence algorithms for educational and research purposes.

## Algorithms Included

1. Particle Swarm Optimization (PSO) ✅
   - CPP ✅  
     - Execution Time: 69.00 ms
     - Result: 0
   - OMP ✅  
     - Execution Time: 2476.00 ms
     - Result: 0
   - THRUST ✅  
     - Execution Time: 11.00 ms
     - Result: 0
   - CUDA ✅  
     - Execution Time: 11.00 ms
     - Result: 0
2. Firefly Algorithm (FA) ✅
   - CPP ✅
     - Execution Time: 16882.00 ms
     - Result: 1.1178274619
   - OMP ✅
     - Execution Time: 17627.00 ms
     - Result: 1.0961358033
   - THRUST ✅
     - Execution Time: 3656.00 ms
     - Result: 1.0304694993
   - CUDA ✅
     - Execution Time: 3659.00 ms
     - Result: 0.5231345270
3. Moth-Flame Optimization (MFO) Algorithm ✅
   - CPP ✅
     - Execution Time: 2620.00 ms
     - Result: 30.8220546814
   - OMP ✅
     - Execution Time: 2666.00 ms
     - Result: 17.1520949785
   - THRUST ✅
     - Execution Time: 31.00 ms
     - Result: 0
   - CUDA ✅
     - Execution Time: 34.00 ms
     - Result: 0
4. Grey Wolf Optimizer (GWO) ✅
   - CPP ✅
     - Execution Time: 145.00 ms
     - Result: 0
   - OMP ✅
     - Execution Time: 244.00 ms
     - Result: 0
   - THRUST ✅
     - Execution Time: 28.00 ms
     - Result: 0
   - CUDA ✅
     - Execution Time: 17.00 ms
     - Result: 0
5. Whale Optimization Algorithm (WOA) ✅
   - CPP ✅
     - Execution Time: 92.00 ms
     - Result: 0
   - OMP ✅
     - Execution Time: 235.00 ms
     - Result: 0
   - THRUST ✅
     - Execution Time: 17.00 ms
     - Result: 0
   - CUDA ✅
     - Execution Time: 17.00 ms
     - Result: 0
6. Artificial Bee Colony (ABC) Algorithm ✅
   - CPP ✅
     - Execution Time: 1871.00 ms
     - Result: 0
   - OMP ✅
     - Execution Time: 413.00
     - Result: 0
   - THRUST ✅
     - Execution Time: 
     - Result: 
   - CUDA ✅
7. Dragonfly Algorithm (DA) ✅
   - CPP ✅
   - OMP ✅
   - THRUST ✅
   - CUDA ✅
8. Salp Swarm Algorithm (SSA) ✅
   - CPP ✅
   - OMP ✅
   - THRUST ✅
   - CUDA ✅
9. Grasshopper Optimization Algorithm (GOA) ✅
   - CPP ✅
   - OMP ✅
   - THRUST ✅
   - CUDA ✅
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