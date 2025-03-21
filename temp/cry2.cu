#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define BLOCK 16
#define DIM 16
#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

extern "C" void launchKernel2(short int *smoothedim, int rows, int cols,
                              short int *delta_x, short int *delta_y);

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

__global__ void GaussianBlurXShared(int center, unsigned char *image, float *kernel, float *tempim, int rows, int cols)
{
    // printf("Hello from CUDA Kernel!\n");
    float dot, /* Dot product summing variable. */
        sum;   /* Sum of the kernel weights variable. */

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    // int tid = threadIdx.x;

    if (r >= rows || c >= cols)
        return;

    extern __shared__ float sharedMemory[];
    float *SharedImg = sharedMemory;
    float *SharedKernel = &SharedImg[blockDim.x];

    if (threadIdx.x < (2 * center + 1))
    {
        SharedKernel[threadIdx.x] = kernel[threadIdx.x];
    }
    if (c < cols)
    {
        SharedImg[threadIdx.x] = (float)image[r * cols + c];
    }
    else
    {
        SharedImg[threadIdx.x] = 0.0f; // Handle out-of-bounds
    }

    __syncthreads(); // Ensure all threads load data before computation

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
            dot += (float)SharedImg[r * cols + (c + cc)] * SharedKernel[center + cc];
            sum += SharedKernel[center + cc];
        }
    }
    tempim[r * cols + c] = dot / sum;
}
__global__ void GaussianBlurYShared(int center, float *kernel, float *tempim, int rows, int cols, short int *smoothedim)
{
    // printf("Hello from CUDA Kernel!\n");
    float dot, /* Dot product summing variable. */
        sum;   /* Sum of the kernel weights variable. */

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y;

    if (r >= rows || c >= cols)
        return;

    extern __shared__ float sharedMemory[];
    float *SharedKernel = sharedMemory;
    float *SharedTemp = &SharedKernel[2 * center + 1];

    if (tid < (2 * center + 1))
    {
        SharedKernel[tid] = kernel[tid];
    }

    if (r < rows)
    {
        SharedTemp[tid] = tempim[r * cols + c];
    }
    else
    {
        SharedTemp[tid] = 0.0f;
    }

    __syncthreads();

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
            dot += SharedTemp[(r + rr) * cols + c] * SharedKernel[center + rr];
            sum += SharedKernel[center + rr];
        }
    }
    smoothedim[r * cols + c] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5);
}

__global__ void derivative_x_y_L1(short int *smoothedim, int rows, int cols,
                                  short int *delta_x)
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
__global__ void derivative_x_y_L2(short int *smoothedim, int rows, int cols,
                                  short int *delta_y)
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

    // c = blockIdx.x * blockDim.x + threadIdx.x;
    // if (c<cols)
    /*for(c=0;c<cols;c++)
    {
        pos = c;
        delta_y[pos] = smoothedim[pos + cols] - smoothedim[pos];
        pos += cols;
        for (r = 1; r < (rows - 1); r++, pos += cols)
        {
            delta_y[pos] = smoothedim[pos + cols] - smoothedim[pos - cols];
        }
        delta_y[pos] = smoothedim[pos] - smoothedim[pos - cols];
    }
    */
}
__global__ void derivative_x_y_L1_Shared(short int *smoothedim, int rows, int cols,
                                         short int *delta_x)
{
    __shared__ short int shared_smoothedim[BLOCK][BLOCK + 2]; // +2 for halo cells

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;

    // Load data into shared memory
    if (row < rows && col < cols)
    {
        shared_smoothedim[ty][tx + 1] = smoothedim[row * cols + col];
    }

    // Load halo cells (left and right boundaries of the tile)
    if (tx == 0 && col > 0)
    {
        shared_smoothedim[ty][0] = smoothedim[row * cols + (col - 1)];
    }
    if (tx == BLOCK - 1 && col < cols - 1)
    {
        shared_smoothedim[ty][BLOCK + 1] = smoothedim[row * cols + (col + 1)];
    }
    __syncthreads();
    if (row < rows && col < cols)
    {
        if (col > 0 && col < cols - 1)
        {
            delta_x[row * cols + col] = shared_smoothedim[ty][tx + 2] - shared_smoothedim[ty][tx];
        }
        else if (col == 0)
        {
            delta_x[row * cols + col] = smoothedim[row * cols + col + 1] - smoothedim[row * cols + col];
        }
        else if (col == cols - 1)
        {
            delta_x[row * cols + col] = smoothedim[row * cols + col] - smoothedim[row * cols + col - 1];
        }
    }
    __syncthreads();
}
__global__ void derivative_x_y_L2_Shared(short int *smoothedim, int rows, int cols,
                                         short int *delta_y)
{
    __shared__ short int shared_smoothedim[BLOCK + 2][BLOCK]; // +2 for halo cells

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;

    // Load data into shared memory
    if (row < rows && col < cols)
    {
        shared_smoothedim[ty + 1][tx] = smoothedim[row * cols + col];
    }

    // Load halo cells (top and bottom boundaries of the tile)
    if (ty == 0 && row > 0)
    {
        shared_smoothedim[0][tx] = smoothedim[(row - 1) * cols + col];
    }
    if (ty == BLOCK - 1 && row < rows - 1)
    {
        shared_smoothedim[BLOCK + 1][tx] = smoothedim[(row + 1) * cols + col];
    }
    __syncthreads();
    if (row < rows && col < cols)
    {
        if (row > 0 && row < rows - 1)
        {
            delta_y[row * cols + col] = shared_smoothedim[ty + 2][tx] - shared_smoothedim[ty][tx];
        }
        else if (row == 0)
        {
            delta_y[row * cols + col] = smoothedim[(row + 1) * cols + col] - smoothedim[row * cols + col];
        }
        else if (row == rows - 1)
        {
            delta_y[row * cols + col] = smoothedim[row * cols + col] - smoothedim[(row - 1) * cols + col];
        }
    }
    __syncthreads();
}

void launchKernel(int center, unsigned char *image, float *kernel, float *tempim, int rows, int cols, short int **smoothedim, int *windowsize)
{
    int size = rows * cols;
    unsigned char *d_image;
    float *d_kernel;
    float *d_tempim;
    short int *d_smoothedim;
    cudaError_t err;
    /*
    center = (*windowsize) / 2;
    */

    float milliseconds = 0;

    // Allocate device memory
    cudaEvent_t memAllocStart, memAllocStop;
    cudaEventCreate(&memAllocStart);
    cudaEventCreate(&memAllocStop);
    cudaEventRecord(memAllocStart);

    cudaMalloc((void **)&d_image, size * sizeof(char));
    cudaMalloc((void **)&d_tempim, size * sizeof(float));
    cudaMalloc((void **)&d_kernel, (*windowsize) * sizeof(float));
    err = cudaMalloc((void **)&d_smoothedim, size * sizeof(short int));
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed for d_smoothedim: %s\n", cudaGetErrorString(err));
        return;
    }
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
    // cudaMemcpy(d_smoothedim, smoothedim, size * sizeof(short int), cudaMemcpyHostToDevice);

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
    /*
    printf("With Shared memory \n");
    int sharedMemSize = ((2 * center + 1) * sizeof(float)) + (BLOCK * sizeof(float));
    GaussianBlurXShared<<<numofblocks, threadperBlock, sharedMemSize>>>(center, d_image, d_kernel, d_tempim, rows, cols);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("blur x Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();

    GaussianBlurYShared<<<numofblocks, threadperBlock, sharedMemSize>>>(center, d_kernel, d_tempim, rows, cols, d_smoothedim);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("blur y Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaDeviceSynchronize();
*/
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

__global__ void kernel()
{
    printf("Hello from CUDA Kernel!\n");
}
void launchKernel2(short int *smoothedim, int rows, int cols, short int *delta_x, short int *delta_y)
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

    printf("Without Shared memory \n");

    derivative_x_y_L1<<<numofblocks, threadperBlock>>>(d_smoothedim, rows, cols, d_delta_x);
    cudaDeviceSynchronize();

    derivative_x_y_L2<<<numofblocks, threadperBlock>>>(d_smoothedim, rows, cols, d_delta_y);
    cudaDeviceSynchronize();

    /*
    printf("With Shared memory 2b\n");
    int sharedMemSize = ((BLOCK + 2) * sizeof(float)) + (BLOCK * sizeof(float));
    derivative_x_y_L1_Shared<<<numofblocks, threadperBlock, sharedMemSize>>>(d_smoothedim, rows, cols, d_delta_x);
    cudaDeviceSynchronize();

    derivative_x_y_L2_Shared<<<numofblocks, threadperBlock, sharedMemSize>>>(d_smoothedim, rows, cols, d_delta_y);
    cudaDeviceSynchronize();
    */

    cudaEventRecord(kernelStop);
    cudaEventSynchronize(kernelStop);
    cudaEventElapsedTime(&milliseconds, kernelStart, kernelStop);
    printf("Kernel Execution Time: %.3f ms\n", milliseconds);

    // Copy data from GPU to CPU
    cudaEvent_t memCopyBackStart, memCopyBackStop;
    cudaEventCreate(&memCopyBackStart);
    cudaEventCreate(&memCopyBackStop);
    cudaEventRecord(memCopyBackStart);

    cudaMemcpy(delta_x, d_delta_x, size * sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_y, d_delta_y, size * sizeof(short int), cudaMemcpyDeviceToHost);
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

/*
//////////////////////////////
// CUDA Kernel for x blurring using shared memory
__global__ void optimized_gaussian_smooth_x_shared(unsigned char *image, float *kernel, int rows, int cols, int center, float *tempim)
{
    extern __shared__ float sharedMem[];
    float *sharedKernel = sharedMem;
    unsigned char *sharedImage = (unsigned char *)&sharedKernel[2 * center + 1];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    double dot = 0.0, sum = 0.0;

    // Load kernel into shared memory
    if (threadIdx.y == 0)
        for (int cc = -center; cc <= center; cc++)
            sharedKernel[center + cc] = kernel[center + cc];

    __syncthreads();

    if (r < rows && c < cols)
    {
        // Load image into shared memory
        sharedImage[threadIdx.y * blockDim.x + threadIdx.x] = image[r * cols + c];
        __syncthreads();

        for (int cc = -center; cc <= center; cc++)
        {
            if ((c + cc) >= 0 && (c + cc) < cols)
            {
                int tx = threadIdx.x + cc;
                if (tx >= 0 && tx < blockDim.x)
                {
                    // Use shared memory when within block bounds
                    dot += (double)sharedImage[threadIdx.y * blockDim.x + tx] * sharedKernel[center + cc];
                }
                else
                {
                    // use global memory when outside
                    dot += (double)image[r * cols + (c + cc)] * sharedKernel[center + cc];
                }
                sum += sharedKernel[center + cc];
            }
        }
        tempim[r * cols + c] = dot / sum;
    }
}

// CUDA Kernel for y blurring using shared memory
__global__ void optimized_gaussian_smooth_y_shared(float *tempim, float *kernel, int rows, int cols, int center, short int *smoothedim)
{
    extern __shared__ float sharedMem[];
    float *sharedKernel = sharedMem;
    float *sharedTempim = &sharedKernel[2 * center + 1];

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    float dot = 0.0, sum = 0.0;

    // Load kernel into shared memory
    if (threadIdx.y == 0)
        for (int cc = -center; cc <= center; cc++)
            sharedKernel[center + cc] = kernel[center + cc];

    __syncthreads();

    if (r < rows && c < cols)
    {
        sharedTempim[threadIdx.y * blockDim.x + threadIdx.x] = tempim[r * cols + c];
        __syncthreads();

        for (int rr = -center; rr <= center; rr++)
        {
            if ((r + rr) >= 0 && (r + rr) < rows)
            {
                int ty = threadIdx.y + rr;
                if (ty >= 0 && ty < blockDim.y)
                {
                    dot += sharedTempim[ty * blockDim.x + threadIdx.x] * sharedKernel[center + rr];
                }
                else
                {
                    dot += tempim[(r + rr) * cols + c] * sharedKernel[center + rr];
                }
                sum += sharedKernel[center + rr];
            }
        }
        smoothedim[r * cols + c] = (short int)(dot * BOOSTBLURFACTOR / sum + 0.5);
    }
}
///////////////////////////////////
_global_ void compute_delta_x_kernel(short int *smoothedim, int rows, int cols, short int *delta_x)
{   _shared_ short int shared_smoothedim[TILE_WIDTH][TILE_WIDTH + 2]; // +2 for halo cells

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Load data into shared memory
    if (row < rows && col < cols) {
        shared_smoothedim[ty][tx + 1] = smoothedim[row * cols + col];
    }

    // Load halo cells (left and right boundaries of the tile)
    if (tx == 0 && col > 0) {
        shared_smoothedim[ty][0] = smoothedim[row * cols + (col - 1)];
    }
    if (tx == TILE_WIDTH - 1 && col < cols - 1) {
        shared_smoothedim[ty][TILE_WIDTH + 1] = smoothedim[row * cols + (col + 1)];
    }

    __syncthreads();

    // Compute delta_x
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
_global_ void compute_delta_y_kernel(short int *smoothedim, int rows, int cols, short int *delta_y) {
    _shared_ short int shared_smoothedim[TILE_WIDTH + 2][TILE_WIDTH]; // +2 for halo cells

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Load data into shared memory
    if (row < rows && col < cols) {
        shared_smoothedim[ty + 1][tx] = smoothedim[row * cols + col];
    }

    // Load halo cells (top and bottom boundaries of the tile)
    if (ty == 0 && row > 0) {
        shared_smoothedim[0][tx] = smoothedim[(row - 1) * cols + col];
    }
    if (ty == TILE_WIDTH - 1 && row < rows - 1) {
        shared_smoothedim[TILE_WIDTH + 1][tx] = smoothedim[(row + 1) * cols + col];
    }

    __syncthreads();

    // Compute delta_y
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
extern "C" void derrivative_x_y_cuda(short int *smoothedim, int rows, int cols, short int **delta_x, short int **delta_y) {
    short int *d_smoothedim, *d_delta_x, *d_delta_y;
    size_t size = rows * cols * sizeof(short int);

    // Allocate memory on the host for delta_x and delta_y
    *delta_x = (short int *)malloc(size);
    *delta_y = (short int *)malloc(size);
    if (*delta_x == NULL || *delta_y == NULL) {
        fprintf(stderr, "Error allocating memory on the host.\n");
        exit(1);
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_smoothedim, size);
    cudaMalloc((void**)&d_delta_x, size);
    cudaMalloc((void**)&d_delta_y, size);

    // Copy smoothedim to the device
    cudaMemcpy(d_smoothedim, smoothedim, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH, (rows + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the delta_x kernel
    compute_delta_x_kernel<<<dimGrid, dimBlock>>>(d_smoothedim, rows, cols, d_delta_x);

    // Launch the delta_y kernel
    compute_delta_y_kernel<<<dimGrid, dimBlock>>>(d_smoothedim, rows, cols, d_delta_y);

    // Copy the results back to the host
    cudaMemcpy(*delta_x, d_delta_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(*delta_y, d_delta_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_smoothedim);
    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
}
    */