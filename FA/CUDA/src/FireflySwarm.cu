#include "FireflySwarm.h"
#include "ObjectiveFunction.h"
#include <iostream>
#include <random>
#include <limits>

__global__ void updateFireflies(Firefly* fireflies, size_t numFireflies, double alpha, double betaMin, double gamma, double searchSpaceMin, double searchSpaceMax) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFireflies) {
        fireflies[idx].updatePosition(fireflies, numFireflies, alpha, betaMin, gamma, searchSpaceMin, searchSpaceMax);
        fireflies[idx].evaluateIntensity(ObjectiveFunction::rosenbrock);
    }
}

FireflySwarm::FireflySwarm(size_t size, double searchSpaceMin, double searchSpaceMax, std::function<double(double, double)> objectiveFunc)
    : objectiveFunc(objectiveFunc), searchSpaceMin(searchSpaceMin), searchSpaceMax(searchSpaceMax), dis(searchSpaceMin, searchSpaceMax) {
    std::random_device rd;
    gen.seed(rd());

    numFireflies = size;

    cudaMalloc(&d_fireflies, size * sizeof(Firefly));
    Firefly* h_fireflies = new Firefly[size];
    for (size_t i = 0; i < size; ++i) {
        h_fireflies[i] = Firefly(randomDouble(), randomDouble());
    }
    cudaMemcpy(d_fireflies, h_fireflies, size * sizeof(Firefly), cudaMemcpyHostToDevice);
    delete[] h_fireflies;
}

FireflySwarm::~FireflySwarm() {
    cudaFree(d_fireflies);
}

void FireflySwarm::optimize(int maxIterations) {
    int blockSize = 256;
    int numBlocks = (numFireflies + blockSize - 1) / blockSize;

    for (int iter = 0; iter < maxIterations; ++iter) {
        updateFireflies<<<numBlocks, blockSize>>>(d_fireflies, numFireflies, alpha, betaMin, gamma, searchSpaceMin, searchSpaceMax);
        cudaDeviceSynchronize();
    }
}

void FireflySwarm::printBestFirefly() const {
    Firefly* h_fireflies = new Firefly[numFireflies];
    cudaMemcpy(h_fireflies, d_fireflies, numFireflies * sizeof(Firefly), cudaMemcpyDeviceToHost);

    auto bestFirefly = std::max_element(h_fireflies, h_fireflies + numFireflies,
                                        [](const Firefly& a, const Firefly& b) {
                                            return a.getIntensityHost() < b.getIntensityHost();
                                        });
    std::cout << "Best Firefly Position: (" << bestFirefly->getPositionHost().first << ", " << bestFirefly->getPositionHost().second << ")\n";
    std::cout << "Best Firefly Intensity: " << bestFirefly->getIntensityHost() << std::endl;

    delete[] h_fireflies;
}

double FireflySwarm::randomDouble() {
    return dis(gen);
}
