#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed with error: " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable devices found." << std::endl;
        return 1;
    }

    int device = 0;
    error_id = cudaSetDevice(device);

    if (error_id != cudaSuccess) {
        std::cerr << "Error: cudaSetDevice failed with error: " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    cudaDeviceProp deviceProp;
    error_id = cudaGetDeviceProperties(&deviceProp, device);

    if (error_id != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceProperties failed with error: " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    std::cout << "sm_" << deviceProp.major << deviceProp.minor << std::endl;

    return 0;
}
