#include <cuda_runtime.h>
#include <stdio.h>
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
    /*
    center = (*windowsize) / 2;
    */
    // Allocate device memory
    cudaMalloc((void **)&d_image, size * sizeof(char));
    cudaMalloc((void **)&d_tempim, size * sizeof(float));
    cudaMalloc((void **)&d_kernel, (*windowsize) * sizeof(float));
    cudaMalloc((void **)&d_smoothedim, size * sizeof(short int));

    // Copy data to GPU
    cudaMemcpy(d_image, image, size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tempim, tempim, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, (*windowsize) * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_smoothedim, smoothedim, size * sizeof(short int), cudaMemcpyHostToDevice);

    int c = (cols + BLOCK - 1) / BLOCK;
    int r = (rows + BLOCK - 1) / BLOCK;
    dim3 numofblocks(c, r);
    dim3 threadperBlock(BLOCK, BLOCK);
    /*
    printf("Without Shared memory\n");
    GaussianBlurX<<<numofblocks, threadperBlock>>>(center, d_image, d_kernel, d_tempim, rows, cols);
    cudaDeviceSynchronize();

    GaussianBlurY<<<numofblocks, threadperBlock>>>(center, d_kernel, d_tempim, rows, cols, d_smoothedim);
    cudaDeviceSynchronize();
    */
    printf("With Shared memory \n");
    int sharedMemSize = ((2 * center + 1) * sizeof(float)) + (BLOCK * sizeof(float));
    GaussianBlurXShared<<<numofblocks, threadperBlock, sharedMemSize>>>(center, d_image, d_kernel, d_tempim, rows, cols);
    cudaDeviceSynchronize();

    GaussianBlurYShared<<<numofblocks, threadperBlock, sharedMemSize>>>(center, d_kernel, d_tempim, rows, cols, d_smoothedim);
    cudaDeviceSynchronize();
    
    // Copy data to GPU
    cudaMemcpy(*smoothedim, d_smoothedim, size * sizeof(short int), cudaMemcpyDeviceToHost);

    // Cuda Free
    cudaFree(d_image);
    cudaFree(d_tempim);
    cudaFree(d_kernel);
    cudaFree(d_smoothedim);
}
__global__ void kernel()
{
    printf("Hello from CUDA Kernel!\n");
}
void launchKernel2(short int *smoothedim, int rows, int cols, short int *delta_x, short int *delta_y)
{
    int size = rows * cols;
    short int *d_delta_x, *d_delta_y, *d_smoothedim;
    cudaMalloc((void **)&d_delta_y, size * sizeof(short int));
    cudaMalloc((void **)&d_delta_x, size * sizeof(short int));
    cudaMalloc((void **)&d_smoothedim, size * sizeof(short int));
    //(short *) malloc(rows*cols* sizeof(short)

    cudaMemcpy(d_smoothedim, smoothedim, size * sizeof(short int), cudaMemcpyHostToDevice);
    int c = (cols + BLOCK - 1) / BLOCK;
    int r = (rows + BLOCK - 1) / BLOCK;
    dim3 numofblocks(c, r);
    dim3 threadperBlock(BLOCK, BLOCK);
    /*
    printf("Without Shared memory 2a\n");

    
    derivative_x_y_L1<<<numofblocks, threadperBlock>>>(d_smoothedim, rows, cols, d_delta_x);
    cudaDeviceSynchronize();

    derivative_x_y_L2<<<numofblocks, threadperBlock>>>(d_smoothedim, rows, cols, d_delta_y);
    cudaDeviceSynchronize();
*/
    printf("With Shared memory 2b\n");
    int sharedMemSize = ((BLOCK + 2) * sizeof(float)) + (BLOCK * sizeof(float));
    derivative_x_y_L1_Shared<<<numofblocks, threadperBlock, sharedMemSize>>>(d_smoothedim, rows, cols, d_delta_x);
    cudaDeviceSynchronize();

    derivative_x_y_L2_Shared<<<numofblocks, threadperBlock, sharedMemSize>>>(d_smoothedim, rows, cols, d_delta_y);
    cudaDeviceSynchronize();

    cudaMemcpy(delta_x, d_delta_x, size * sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_y, d_delta_y, size * sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaFree(d_smoothedim);
}
