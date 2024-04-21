# Swarm Intelligence Algorithms

Welcome to the Swarm Intelligence Algorithms Repository! This repository contains implementations of various swarm intelligence algorithms in C++, OpenMP, CUDA, and Thrust. The goal of this repository is to provide a comprehensive collection of swarm intelligence algorithms for educational and research purposes.

## Algorithms Included

1. Particle Swarm Optimization (PSO) ✅
   - TSP:
     - CPP
       - Execution Time:
       - Result:
     - OMP
       - Execution Time:
       - Result:
     - THRUST
       - Execution
       - Result:
     - CUDA
       - Execution Time:
       - Result:
   - None-TSP: ✅
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
   - TSP:
     - CPP
       - Execution Time:
       - Result:
     - OMP
       - Execution Time:
       - Result:
     - THRUST
       - Execution
       - Result:
     - CUDA
       - Execution Time:
       - Result:
   - None-TSP: ✅
     - CPP ✅
       - Execution Time:
       - Result:
     - OMP ✅
       - Execution Time:
       - Result:
     - THRUST ✅
       - Execution Time:
       - Result:
     - CUDA ✅
       - Execution Time:
       - Result:
7. Dragonfly Algorithm (DA) ✅
   - CPP ✅
     - Execution Time: 125.00 ms
     - Result: 2.0665598224
   - OMP ✅
     - Execution Time: 116.00 ms
     - Result: 1.8372467783
   - THRUST ✅
     - Execution Time: 13.00 ms
     - Result: 0.0089122206
   - CUDA ✅
     - Execution Time: 14.00 ms
     - Result: 1.4157596691
8. Salp Swarm Algorithm (SSA) ✅
   - CPP ✅
     - Execution Time: 50.00 ms
     - Result: 0.0019778109
   - OMP ✅
     - Execution Time: 2124.00 ms
     - Result: 0.0000558706
   - THRUST ✅
     - Execution Time: 13.00 ms
     - Result: 0.0168243996
   - CUDA ✅
     - Execution Time: 14.00 ms
     - Result: 1.0017446653
9. Grasshopper Optimization Algorithm (GOA) ✅
   - CPP ✅
     - Execution Time: 17589.00 ms
     - Result: 0.0583486423
   - OMP ✅
     - Execution Time: 21697.00 ms
     - Result: 0.0037905337
   - THRUST ✅
     - Execution Time: 970.00 ms
     - Result: 0.0018583121
   - CUDA ✅
     - Execution Time: 979.00 ms
     - Result: 0.0628388407
10. Ant Colony Optimization (ACO) ✅
    - TSP:
      - CPP
        - Execution Time:
        - Result:
      - OMP
        - Execution Time:
        - Result:
      - THRUST
        - Execution
        - Result:
      - CUDA
        - Execution Time:
        - Result:
    - None-TSP: ✅
      - CPP ✅
        - Execution Time:
        - Result:
      - OMP ✅
        - Execution Time:
        - Result:
      - THRUST ✅
        - Execution Time:
        - Result:
      - CUDA ✅
        - Execution Time:
        - Result:

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

## License

This repository is licensed under the [MIT License](LICENSE). Feel free to use the code for educational and research purposes. If you use any of the implementations in your work, please consider citing this repository.