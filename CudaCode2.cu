#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define BLOCK 16
#define DIM 16
#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

extern "C" void launchKernel2(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y);
extern "C" void launchKernel(int center, unsigned char *image, float *kernel, float *tempim, int rows, int cols, short int **smoothedim, int *windowsize);

__global__ void GaussianBlurXShared(int center, unsigned char *image, float *kernel, float *tempim, int rows, int cols)
{
    extern __shared__ float sharedMemory[];
    float *SharedImg = sharedMemory;
    float *SharedKernel = (float *)&SharedImg[blockDim.x * blockDim.y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int c = bx * blockDim.x + tx;
    int r = by * blockDim.y + ty;

    if (r >= rows || c >= cols)
        return;

    // Load kernel into shared memory
    if (tx < (2 * center + 1))
    {
        SharedKernel[tx] = kernel[tx];
    }

    // Load image into shared memory
    SharedImg[ty * blockDim.x + tx] = (float)image[r * cols + c];

    __syncthreads();

    // Compute the blurred value
    float dot = 0.0f, sum = 0.0f;
    for (int cc = -center; cc <= center; cc++)
    {
        int col = c + cc;
        if (col >= 0 && col < cols)
        {
            dot += SharedImg[ty * blockDim.x + (tx + cc)] * SharedKernel[center + cc];
            sum += SharedKernel[center + cc];
        }
    }

    tempim[r * cols + c] = dot / sum;
}
__global__ void GgaussianBlurYShared(int center, float *kernel, float *tempim, int rows, int cols, short int *smoothedim)
{
    extern __shared__ float sharedMemory[];
    float *SharedImg = sharedMemory;
    float *SharedKernel = (float *)&SharedImg[blockDim.x * blockDim.y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int c = bx * blockDim.x + tx;
    int r = by * blockDim.y + ty;

    if (r >= rows || c >= cols)
        return;

    // Load kernel into shared memory
    if (ty < (2 * center + 1))
    {
        SharedKernel[ty] = kernel[ty];
    }

    // Load tempim into shared memory
    SharedImg[ty * blockDim.x + tx] = tempim[r * cols + c];

    __syncthreads();

    // Compute the blurred value
    float dot = 0.0f, sum = 0.0f;
    for (int rr = -center; rr <= center; rr++)
    {
        int row = r + rr;
        if (row >= 0 && row < rows)
        {
            dot += SharedImg[(ty + rr) * blockDim.x + tx] * SharedKernel[center + rr];
            sum += SharedKernel[center + rr];
        }
    }

    smoothedim[r * cols + c] = (short int)(dot / sum);
}
__global__ void derivative_x_y_L1_Shared(short int *smoothedim, int rows, int cols,
                                         short int *delta_x)
{
    __shared__ short int shared_smoothedim[BLOCK][BLOCK + 2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;

    if (row < rows && col < cols) {
        shared_smoothedim[ty][tx + 1] = smoothedim[row * cols + col];
    }

    if (tx == 0 && col > 0) {
        shared_smoothedim[ty][0] = smoothedim[row * cols + (col - 1)];
    }
    if (tx == BLOCK - 1 && col < cols - 1) {
        shared_smoothedim[ty][BLOCK + 1] = smoothedim[row * cols + (col + 1)];
    }

    __syncthreads();

    if (row < rows && col < cols) {
        if (col > 0 && col < cols - 1) {
            delta_x[row * cols + col] = shared_smoothedim[ty][tx + 2] - shared_smoothedim[ty][tx];
        } else if (col == 0) {
            delta_x[row * cols + col] = smoothedim[row * cols + col + 1] - smoothedim[row * cols + col];
        } else if (col == cols - 1) {
            delta_x[row * cols + col] = smoothedim[row * cols + col] - smoothedim[row * cols + col - 1];
        }
    }
}
__global__ void derivative_x_y_L2_Shared(short int *smoothedim, int rows, int cols,
                                         short int *delta_y)
{
    __shared__ short int shared_smoothedim[BLOCK + 2][BLOCK];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;

    if (row < rows && col < cols) {
        shared_smoothedim[ty + 1][tx] = smoothedim[row * cols + col];
    }

    if (ty == 0 && row > 0) {
        shared_smoothedim[0][tx] = smoothedim[(row - 1) * cols + col];
    }
    if (ty == BLOCK - 1 && row < rows - 1) {
        shared_smoothedim[BLOCK + 1][tx] = smoothedim[(row + 1) * cols + col];
    }

    __syncthreads();

    if (row < rows && col < cols) {
        if (row > 0 && row < rows - 1) {
            delta_y[row * cols + col] = shared_smoothedim[ty + 2][tx] - shared_smoothedim[ty][tx];
        } else if (row == 0) {
            delta_y[row * cols + col] = smoothedim[(row + 1) * cols + col] - smoothedim[row * cols + col];
        } else if (row == rows - 1) {
            delta_y[row * cols + col] = smoothedim[row * cols + col] - smoothedim[(row - 1) * cols + col];
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

    cudaMalloc((void **)&d_image, size * sizeof(unsigned char));
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

    cudaMemcpy(d_image, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_tempim, tempim, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, (*windowsize) * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

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

    printf("With Shared memory \n");
    size_t sharedMemSizeX = (BLOCK * BLOCK) * sizeof(unsigned char) + (2 * center + 1) * sizeof(float);
    size_t sharedMemSizeY = (BLOCK * BLOCK) * sizeof(float) + (2 * center + 1) * sizeof(float);

    // size_t sharedMemSizeX = (2 * center + 1) * sizeof(float) + (numofblocks.x * numofblocks.y) * sizeof(unsigned char);
    GaussianBlurXShared<<<numofblocks, threadperBlock, sharedMemSizeX>>>(center, d_image, d_kernel, d_tempim, rows, cols);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("blur x Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("aarij\n");
    // size_t sharedMemSizeY = (2 * center + 1) * sizeof(float) + (numofblocks.x * numofblocks.y) * sizeof(float);
    GgaussianBlurYShared<<<numofblocks, threadperBlock, sharedMemSizeY>>>(center, d_kernel, d_tempim, rows, cols, d_smoothedim);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("blurrrrrrrrrrrrrrrrrrrrrrr y Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    // cudaDeviceSynchronize();

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    cudaEventElapsedTime(&milliseconds, kernelStart, kernelStop);
    printf("Kernel Execution Time: %.3f ms\n", milliseconds);

    // Copy data to GPU
    cudaEvent_t memCopyBackStart, memCopyBackStop;
    cudaEventCreate(&memCopyBackStart);
    cudaEventCreate(&memCopyBackStop);
    cudaEventRecord(memCopyBackStart);

    cudaMemcpy(*smoothedim, d_smoothedim, size * sizeof(short int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

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
    // CUDA Events for Timing
    float milliseconds = 0;

    // Allocate device memory
    cudaEvent_t memAllocStart, memAllocStop;
    cudaEventCreate(&memAllocStart);
    cudaEventCreate(&memAllocStop);
    cudaEventRecord(memAllocStart);

    cudaMalloc((void **)&d_delta_y, size * sizeof(short int));
    cudaMalloc((void **)&d_delta_x, size * sizeof(short int));
    cudaMalloc((void **)&d_smoothedim, size * sizeof(short int));
    //(short *) malloc(rows*cols* sizeof(short)
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

    printf("With Shared memory 2b\n");
    int sharedMemSize = ((BLOCK + 2) * sizeof(float)) + (BLOCK * sizeof(float));
    derivative_x_y_L1_Shared<<<numofblocks, threadperBlock, sharedMemSize>>>(d_smoothedim, rows, cols, d_delta_x);
    cudaDeviceSynchronize();

    derivative_x_y_L2_Shared<<<numofblocks, threadperBlock, sharedMemSize>>>(d_smoothedim, rows, cols, d_delta_y);
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
