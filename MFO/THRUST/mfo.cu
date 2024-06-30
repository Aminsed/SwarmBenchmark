#include "ObjectiveFunction.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

struct Moth {
    double position[DIMENSIONS];
    double fitness;
};

struct Flame {
    double position[DIMENSIONS];
};

__device__ void updateFlame(Moth* m, Flame* flame, double* bestFitness) {
    double fitness = objectiveFunction(m->position);
    if (fitness < objectiveFunction(flame->position)) {
        for (int i = 0; i < DIMENSIONS; i++) {
            flame->position[i] = m->position[i];
        }
    }
    if (fitness < *bestFitness) {
        *bestFitness = fitness;
    }
}

__global__ void initializeMoths(Moth* moths, Flame* flames, int* flameIndexes, curandState* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_MOTHS) {
        Moth* m = &moths[tid];
        Flame* f = &flames[tid];
        curandState* s = &state[tid];
        curand_init(clock64(), tid, 0, s);
        for (int i = 0; i < DIMENSIONS; i++) {
            m->position[i] = curand_uniform_double(s) * 10.0 - 5.0;
            f->position[i] = m->position[i];
        }
        m->fitness = objectiveFunction(m->position);
        flameIndexes[tid] = tid;
    }
}

__global__ void updateMoths(Moth* moths, Flame* flames, int* flameIndexes, curandState* state, int iter, double* bestFitness) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_MOTHS) {
        Moth* m = &moths[tid];
        curandState* s = &state[tid];
        int flameIndex = flameIndexes[tid];
        Flame* flame = &flames[flameIndex];
        for (int i = 0; i < DIMENSIONS; i++) {
            double t = (double)iter / MAX_ITERATIONS;
            double r = curand_uniform_double(s);
            double b = 1.0;
            double distance = fabs(flame->position[i] - m->position[i]);
            if (r < 0.5) {
                m->position[i] = distance * exp(b * t) * cos(t * 2 * M_PI) + flame->position[i];
            } else {
                m->position[i] = distance * exp(b * t) * sin(t * 2 * M_PI) + flame->position[i];
            }
        }
        m->fitness = objectiveFunction(m->position);
        updateFlame(m, flame, bestFitness);
    }
}

__global__ void sortMothsByFitness(Moth* moths, int* flameIndexes) {
    for (int i = 0; i < NUM_MOTHS - 1; i++) {
        for (int j = 0; j < NUM_MOTHS - i - 1; j++) {
            if (moths[j].fitness > moths[j + 1].fitness) {
                Moth temp = moths[j];
                moths[j] = moths[j + 1];
                moths[j + 1] = temp;

                int tempIndex = flameIndexes[j];
                flameIndexes[j] = flameIndexes[j + 1];
                flameIndexes[j + 1] = tempIndex;
            }
        }
    }
}

void runMFO(thrust::device_vector<Moth>& d_moths, thrust::device_vector<Flame>& d_flames, thrust::device_vector<int>& d_flameIndexes, thrust::device_vector<curandState>& d_state, thrust::device_vector<double>& d_bestFitness) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_MOTHS + block.x - 1) / block.x);
    std::ofstream outputFile("results.txt");
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        updateMoths<<<grid, block>>>(thrust::raw_pointer_cast(d_moths.data()), thrust::raw_pointer_cast(d_flames.data()), thrust::raw_pointer_cast(d_flameIndexes.data()), thrust::raw_pointer_cast(d_state.data()), iter, thrust::raw_pointer_cast(d_bestFitness.data()));
        cudaDeviceSynchronize();
        sortMothsByFitness<<<1, 1>>>(thrust::raw_pointer_cast(d_moths.data()), thrust::raw_pointer_cast(d_flameIndexes.data()));
        cudaDeviceSynchronize();
        double currentBestFitness;
        cudaMemcpy(&currentBestFitness, thrust::raw_pointer_cast(d_bestFitness.data()), sizeof(double), cudaMemcpyDeviceToHost);
        outputFile << iter + 1 << ": " << currentBestFitness << std::endl;
    }
    outputFile.close();
}

void printResults(thrust::host_vector<Flame>& h_flames, double* bestFitness, double executionTime) {
    std::cout << std::fixed << std::setprecision(10);
    if (DIMENSIONS == 1) {
        std::cout << "Best Flame Position: " << h_flames[0].position[0] << std::endl;
    } else {
        std::cout << "Best Flame Position: (";
        for (int i = 0; i < DIMENSIONS; i++) {
            std::cout << h_flames[0].position[i];
            if (i < DIMENSIONS - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    std::cout << "Best Flame Fitness: " << *bestFitness << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Execution Time: " << executionTime << " milliseconds" << std::endl;
}


int main() {
    // Add error checking for CUDA operations
    cudaError_t cudaStatus;

    thrust::device_vector<Moth> d_moths(NUM_MOTHS);
    thrust::device_vector<Flame> d_flames(NUM_MOTHS);
    thrust::device_vector<int> d_flameIndexes(NUM_MOTHS);
    thrust::device_vector<curandState> d_state(NUM_MOTHS);
    thrust::device_vector<double> d_bestFitness(1);

    double initialBestFitness = std::numeric_limits<double>::max();
    thrust::fill(d_bestFitness.begin(), d_bestFitness.end(), initialBestFitness);

    auto start = std::chrono::high_resolution_clock::now();

    initializeMoths<<<(NUM_MOTHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_moths.data()), 
        thrust::raw_pointer_cast(d_flames.data()), 
        thrust::raw_pointer_cast(d_flameIndexes.data()), 
        thrust::raw_pointer_cast(d_state.data())
    );
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    runMFO(d_moths, d_flames, d_flameIndexes, d_state, d_bestFitness);

    auto end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    thrust::host_vector<Flame> h_flames = d_flames;
    double hostBestFitness;
    cudaStatus = cudaMemcpy(&hostBestFitness, thrust::raw_pointer_cast(d_bestFitness.data()), sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    printResults(h_flames, &hostBestFitness, executionTime);
    return 0;
}