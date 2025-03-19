#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void launchKernel();

__global__ void kernel() {
    printf("Hello from CUDA Kernel!\n");
}

void launchKernel() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
