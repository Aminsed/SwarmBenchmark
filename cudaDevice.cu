#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nDevice " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "CUDA Capability Major/Minor version number: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total amount of global memory: " << deviceProp.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "Maximum number of threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum sizes of each dimension of a block: " << deviceProp.maxThreadsDim[0] << " x "
                  << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "Maximum sizes of each dimension of a grid: " << deviceProp.maxGridSize[0] << " x "
                  << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "Maximum memory pitch: " << deviceProp.memPitch / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Texture alignment: " << deviceProp.textureAlignment << " bytes" << std::endl;
    }

    return 0;
}
