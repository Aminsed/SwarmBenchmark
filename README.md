# Swarm Intelligence Algorithms Repository

Welcome to the Swarm Intelligence Algorithms Repository! This repository contains implementations of various swarm intelligence algorithms in C++, OpenMP, CUDA and Thrust. The goal of this repository is to provide a comprehensive collection of swarm intelligence algorithms for educational and research purposes.

## Algorithms Included

1. Particle Swarm Optimization (PSO) ✅
   - CPP ✅
   - OMP ✅
   - THRUST ✅
   - CUDA ✅
2. Firefly Algorithm (FA)
   - CPP 
   - OMP 
   - THRUST
   - CUDA
3. Moth-Flame Optimization (MFO) Algorithm
   - CPP 
   - OMP 
   - THRUST
   - CUDA
4. Grey Wolf Optimizer (GWO)
   - CPP 
   - OMP 
   - THRUST
   - CUDA
5. Whale Optimization Algorithm (WOA)
   - CPP
   - OMP
   - CUDA
   - THRUST
6. Artificial Bee Colony (ABC) Algorithm
   - CPP 
   - OMP 
   - THRUST
   - CUDA
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
- `CUDA`: Contains the CUDA implementation of the algorithm for GPU acceleration.
- `THRUST`: Contains the THRUST implementation of the algorithm for GPU acceleration.

Inside each subfolder, you will find the respective code files and a README file with instructions on how to compile and run the code.

## Getting Started

To get started with any of the algorithms, navigate to the desired algorithm's folder and choose the implementation you want to use (CPP, OMP, or CUDA).

## Compiling Pure C++ Code

When compiling pure C++ code, you can use a C++ compiler such as g++ (GNU Compiler Collection) or clang++ (Clang C++ Compiler). The compilation process is relatively straightforward.

To compile your C++ code, use the following command:
```
g++ -o myProgram myProgram.cpp -O3
```
or
```
clang++ -o myProgram myProgram.cpp -O3
```
Replace `myProgram.cpp` with the actual name of your C++ code file.

The `-O3` flag is an optimization flag that tells the compiler to perform aggressive optimizations on the code. It can help improve the performance of your program.

If your code consists of multiple source files, you can compile them together by listing all the source files in the compilation command:
```
g++ -o myProgram myProgram.cpp helper.cpp -O3
```
or
```
clang++ -o myProgram myProgram.cpp helper.cpp -O3
```

After successful compilation, you can run your program using:
```
./myProgram
```

## Compiling OpenMP Code

OpenMP is an API for parallel programming in C++. It allows you to write parallel code using compiler directives, making it easier to parallelize your code across multiple CPU cores.

To compile OpenMP code, you need to use a compiler that supports OpenMP, such as g++ or clang++. Additionally, you need to include the appropriate OpenMP flags during compilation.

To compile your OpenMP code using g++, use the following command:
```
g++ -o myProgram myProgram.cpp -fopenmp -O3
```
or with clang++:
```
clang++ -o myProgram myProgram.cpp -fopenmp -O3
```
Replace `myProgram.cpp` with the actual name of your OpenMP code file.

The `-fopenmp` flag tells the compiler to enable OpenMP support and link against the OpenMP runtime library.

If your code consists of multiple source files, you can compile them together by listing all the source files in the compilation command:
```
g++ -o myProgram myProgram.cpp helper.cpp -fopenmp -O3
```
or
```
clang++ -o myProgram myProgram.cpp helper.cpp -fopenmp -O3
```

After successful compilation, you can run your OpenMP program using:
```
./myProgram
```

By default, OpenMP will use all available CPU cores for parallel execution. You can control the number of threads used by setting the `OMP_NUM_THREADS` environment variable before running your program:
```
export OMP_NUM_THREADS=4
./myProgram
```
This sets the number of threads to 4, meaning your program will use 4 CPU cores for parallel execution.

## Compiling CUDA/THRUST Code

When compiling CUDA or THRUST code, it's important to ensure that the code is compiled for the correct GPU architecture. Different GPU architectures have different compute capabilities, and compiling your code with the appropriate architecture flag can significantly impact performance and compatibility.

### Determining GPU Architecture

To determine the GPU architecture of your CUDA-capable device, you can use the `cudaDevice.cu` code provided in this repository. This code queries the device properties and displays important information such as the compute capability, global memory size, shared memory size, and more.

To compile and run `cudaDevice.cu`, follow these steps:

1. Compile the `cudaDevice.cu` code using the following command:
   ```
   nvcc -o cudaDevice cudaDevice.cu
   ```
2. Run the compiled executable:
   ```
   ./cudaDevice
   ```

The output will provide details about your CUDA-capable device(s). Look for the line that says "CUDA Capability Major/Minor version number" to determine the compute capability of your GPU.

For example, if the output shows:
```
CUDA Capability Major/Minor version number: 7.5
```
It means your GPU has a compute capability of 7.5.

### Compiling with the Correct Architecture Flag

Once you know the compute capability of your GPU, you can compile your CUDA or THRUST code with the appropriate architecture flag. The architecture flag ensures that the code is optimized for your specific GPU architecture.

Replace `myProgram.cu` with the actual name of your CUDA or THRUST code file, and replace `XX` with the compute capability of your GPU.

For example, if your GPU has a compute capability of 7.5, you would compile your code using:
```
nvcc -o myProgram myProgram.cu -arch=sm_75
```

It's important to note that compiling with the correct architecture flag is crucial for several reasons:
1. Performance: Compiling with the appropriate architecture flag allows the compiler to optimize the code for your specific GPU architecture, resulting in better performance.
2. Compatibility: Different GPU architectures have different features and capabilities. Compiling with the correct flag ensures that your code is compatible with your GPU and can utilize its available features.
3. Debugging: If you encounter any issues or errors while running your CUDA or THRUST code, compiling with the correct architecture flag can help in identifying and resolving those issues.

By using the `cudaDevice.cu` code to determine your GPU architecture and compiling your code with the appropriate architecture flag, you can ensure optimal performance and compatibility when running swarm intelligence algorithms on CUDA or THRUST.

## Contributing

Contributions to this repository are welcome! If you have an implementation of a swarm intelligence algorithm that is not currently included, or if you want to improve an existing implementation, please submit a pull request. Make sure to follow the repository's structure and provide clear instructions on how to compile and run your code.

## License

This repository is licensed under the [MIT License](LICENSE). Feel free to use the code for educational and research purposes. If you use any of the implementations in your work, please consider citing this repository.