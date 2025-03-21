#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void launchKernel2();

_global_ void kernel() {
    printf("Hello from CUDA Kernel!\n");
}

void launchKernel2() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}