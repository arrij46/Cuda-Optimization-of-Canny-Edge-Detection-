#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define BLOCK 16
#define DIM 16
#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

extern "C" void launchKernel2(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y);
extern "C" void launchKernel(int center, unsigned char *image, float *kernel, float *tempim, int rows, int cols, short int **smoothedim, int *windowsize);

__global__ void GaussianBlurX(int center, unsigned char *image, float *kernel, float *tempim, int rows, int cols)
{
    // printf("Hello from CUDA Kernel!\n");
    float dot, /* Dot product summing variable. */
        sum;   /* Sum of the kernel weights variable. */

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows || c >= cols)
        return;

    /****************************************************************************
     * Blur in the x - direction.
     ****************************************************************************/
    if (VERBOSE)
        printf("   Bluring the image in the X-direction.\n");

    dot = 0.0;
    sum = 0.0;
    for (int cc = (-center); cc <= center; cc++)
    {
        if (((c + cc) >= 0) && ((c + cc) < cols))
        {
            dot += (float)image[r * cols + (c + cc)] * kernel[center + cc];
            sum += kernel[center + cc];
        }
    }
    tempim[r * cols + c] = dot / sum;
}
__global__ void GaussianBlurY(int center, float *kernel, float *tempim, int rows, int cols, short int *smoothedim)
{
    // printf("Hello from CUDA Kernel!\n");
    float dot, /* Dot product summing variable. */
        sum;   /* Sum of the kernel weights variable. */

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows || c >= cols)
        return;
    /****************************************************************************
     * Blur in the y - direction.
     ****************************************************************************/
    if (VERBOSE)
        printf("   Bluring the image in the Y-direction.\n");

    sum = 0.0;
    dot = 0.0;
    for (int rr = (-center); rr <= center; rr++)
    {
        if (((r + rr) >= 0) && ((r + rr) < rows))
        {
            dot += tempim[(r + rr) * cols + c] * kernel[center + rr];
            sum += kernel[center + rr];
        }
    }
    smoothedim[r * cols + c] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5);
}

__global__ void derivative_x_y_L1(short int *smoothedim, int rows, int cols, short int *delta_x)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols)
    {
        if (c > 0 && c < cols - 1)
        {
            delta_x[r * cols + c] = smoothedim[r * cols + (c + 1)] - smoothedim[r * cols + (c - 1)];
        }
        else if (c == 0)
        {
            delta_x[r * cols + c] = smoothedim[r * cols + (c + 1)] - smoothedim[r * cols + c];
        }
        else if (c == cols - 1)
        {
            delta_x[r * cols + c] = smoothedim[r * cols + c] - smoothedim[r * cols + (c - 1)];
        }
    }

    // for(r=0;r<rows;r++)

    /*{
        pos = r * cols;
        delta_x[pos] = smoothedim[pos + 1] - smoothedim[pos];
        pos++;
        for (c = 1; c < (cols - 1); c++, pos++)
        {
            delta_x[pos] = smoothedim[pos + 1] - smoothedim[pos - 1];
        }
        delta_x[pos] = smoothedim[pos] - smoothedim[pos - 1];
    }
    */
}
__global__ void derivative_x_y_L2(short int *smoothedim, int rows, int cols, short int *delta_y)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols)
    {
        if (r > 0 && r < rows - 1)
        {
            delta_y[r * cols + c] = smoothedim[(r + 1) * cols + c] - smoothedim[(r - 1) * cols + c];
        }
        else if (r == 0)
        {
            delta_y[r * cols + c] = smoothedim[(r + 1) * cols + c] - smoothedim[r * cols + c];
        }
        else if (r == rows - 1)
        {
            delta_y[r * cols + c] = smoothedim[r * cols + c] - smoothedim[(r - 1) * cols + c];
        }
    }
}

void launchKernel(int center, unsigned char *image, float *kernel, float *tempim, int rows, int cols, short int **smoothedim, int *windowsize)
{
    int size = rows * cols;
    unsigned char *d_image;
    float *d_kernel;
    float *d_tempim;
    short int *d_smoothedim;
    cudaError_t err;
    float milliseconds = 0;

    // Allocate device memory
    cudaEvent_t memAllocStart, memAllocStop;
    cudaEventCreate(&memAllocStart);
    cudaEventCreate(&memAllocStop);
    cudaEventRecord(memAllocStart);

    cudaMalloc((void **)&d_image, size * sizeof(char));
    cudaMalloc((void **)&d_tempim, size * sizeof(float));
    cudaMalloc((void **)&d_kernel, (*windowsize) * sizeof(float));
    cudaMalloc((void **)&d_smoothedim, size * sizeof(short int));

    cudaEventRecord(memAllocStop);
    cudaEventSynchronize(memAllocStop);
    cudaEventElapsedTime(&milliseconds, memAllocStart, memAllocStop);
    printf("Memory Allocation Time: %.3f ms\n", milliseconds);

    // Copy data to GPU
    cudaEvent_t memCopyStart, memCopyStop;
    cudaEventCreate(&memCopyStart);
    cudaEventCreate(&memCopyStop);
    cudaEventRecord(memCopyStart);

    cudaMemcpy(d_image, image, size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tempim, tempim, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, (*windowsize) * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(memCopyStop);
    cudaEventSynchronize(memCopyStop);
    cudaEventElapsedTime(&milliseconds, memCopyStart, memCopyStop);
    printf("Memory Copy (Host to Device) Time: %.3f ms\n", milliseconds);

    int c = (cols + BLOCK - 1) / BLOCK;
    int r = (rows + BLOCK - 1) / BLOCK;
    dim3 numofblocks(c, r);
    dim3 threadperBlock(BLOCK, BLOCK);

    // Measure Kernel Execution Time
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    cudaEventRecord(kernelStart);

    printf("Without Shared memory\n");

    GaussianBlurX<<<numofblocks, threadperBlock>>>(center, d_image, d_kernel, d_tempim, rows, cols);
    cudaDeviceSynchronize();

    GaussianBlurY<<<numofblocks, threadperBlock>>>(center, d_kernel, d_tempim, rows, cols, d_smoothedim);
    cudaDeviceSynchronize();

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    cudaEventElapsedTime(&milliseconds, kernelStart, kernelStop);
    printf("Kernel Execution Time: %.3f ms\n", milliseconds);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy data to GPU
    cudaEvent_t memCopyBackStart, memCopyBackStop;
    cudaEventCreate(&memCopyBackStart);
    cudaEventCreate(&memCopyBackStop);
    cudaEventRecord(memCopyBackStart);

    cudaMemcpy(*smoothedim, d_smoothedim, size * sizeof(short int), cudaMemcpyDeviceToHost);

    cudaEventRecord(memCopyBackStop);
    cudaEventSynchronize(memCopyBackStop);
    cudaEventElapsedTime(&milliseconds, memCopyBackStart, memCopyBackStop);
    printf("Memory Copy (Device to Host) Time: %.3f ms\n", milliseconds);

    // Cuda Free

    cudaFree(d_image);
    cudaFree(d_tempim);
    cudaFree(d_kernel);
    cudaFree(d_smoothedim);

    cudaEventDestroy(memAllocStart);
    cudaEventDestroy(memAllocStop);
    cudaEventDestroy(memCopyStart);
    cudaEventDestroy(memCopyStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    cudaEventDestroy(memCopyBackStart);
    cudaEventDestroy(memCopyBackStop);
}

void launchKernel2(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y)
{
    int size = rows * cols;
    short int *d_delta_x, *d_delta_y, *d_smoothedim;
    float milliseconds = 0;

    // Allocate device memory
    cudaEvent_t memAllocStart, memAllocStop;
    cudaEventCreate(&memAllocStart);
    cudaEventCreate(&memAllocStop);
    cudaEventRecord(memAllocStart);

    cudaMalloc((void **)&d_delta_y, size * sizeof(short int));
    cudaMalloc((void **)&d_delta_x, size * sizeof(short int));
    cudaMalloc((void **)&d_smoothedim, size * sizeof(short int));

    cudaEventRecord(memAllocStop);
    cudaEventSynchronize(memAllocStop);
    cudaEventElapsedTime(&milliseconds, memAllocStart, memAllocStop);
    printf("Memory Allocation Time: %.3f ms\n", milliseconds);

    // Copy data to GPU
    cudaEvent_t memCopyStart, memCopyStop;
    cudaEventCreate(&memCopyStart);
    cudaEventCreate(&memCopyStop);
    cudaEventRecord(memCopyStart);

    cudaMemcpy(d_smoothedim, smoothedim, size * sizeof(short int), cudaMemcpyHostToDevice);

    cudaEventRecord(memCopyStop);
    cudaEventSynchronize(memCopyStop);
    cudaEventElapsedTime(&milliseconds, memCopyStart, memCopyStop);
    printf("Memory Copy (Host to Device) Time: %.3f ms\n", milliseconds);

    int c = (cols + BLOCK - 1) / BLOCK;
    int r = (rows + BLOCK - 1) / BLOCK;
    dim3 numofblocks(c, r);
    dim3 threadperBlock(BLOCK, BLOCK);

    // Measure Kernel Execution Time
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    cudaEventRecord(kernelStart);

    printf("Without Shared memory 2a\n");

    derivative_x_y_L1<<<numofblocks, threadperBlock>>>(d_smoothedim, rows, cols, d_delta_x);
    cudaDeviceSynchronize();

    derivative_x_y_L2<<<numofblocks, threadperBlock>>>(d_smoothedim, rows, cols, d_delta_y);
    cudaDeviceSynchronize();

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    cudaEventElapsedTime(&milliseconds, kernelStart, kernelStop);
    printf("Kernel Execution Time: %.3f ms\n", milliseconds);

    // Copy data from GPU to CPU
    cudaEvent_t memCopyBackStart, memCopyBackStop;
    cudaEventCreate(&memCopyBackStart);
    cudaEventCreate(&memCopyBackStop);
    cudaEventRecord(memCopyBackStart);

    cudaMemcpy(*delta_x, d_delta_x, size * sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*delta_y, d_delta_y, size * sizeof(short int), cudaMemcpyDeviceToHost);

    cudaEventRecord(memCopyBackStop);
    cudaEventSynchronize(memCopyBackStop);
    cudaEventElapsedTime(&milliseconds, memCopyBackStart, memCopyBackStop);
    printf("Memory Copy (Device to Host) Time: %.3f ms\n", milliseconds);

    // Cuda Free
    cudaEventDestroy(memAllocStart);
    cudaEventDestroy(memAllocStop);
    cudaEventDestroy(memCopyStart);
    cudaEventDestroy(memCopyStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    cudaEventDestroy(memCopyBackStart);
    cudaEventDestroy(memCopyBackStop);

    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaFree(d_smoothedim);
}
