/*
This is an example kernel for perfoming the addition of two 1D vectors.

Compile example:
nvcc -O3 vec_add.cu -o vec_add

version 1 creates a fixed grid with each thread performing potentially
more than one operation
./vec_add 1

version 2 creates a dynamic grid with total number of threads created 
proportional to the number of elements in the vector
./vec_add 2

*/


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "common.h"

// ---------------------------------------------------------------------------------------
// CPU reference code

void vec_add_cpu(float *z, const float *x, const float *y, int N) {
    for (int i = 0; i < N; ++i)
        z[i] = x[i] + y[i];
}

// ---------------------------------------------------------------------------------------
// GPU kernels

__global__ void vec_add_kernel1(float *z, const float *x, const float *y, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        z[i] = x[i] + y[i];
    }
}

__global__ void vec_add_kernel2(float *z, const float *x, const float *y, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
        z[index] = x[index] + y[index];
}


// ---------------------------------------------------------------------------------------
// Kernel launchers

void vec_add1(float *z, const float *x, const float *y, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    vec_add_kernel1<<<grid_size, block_size>>>(z, x, y, N);
    cudaCheck(cudaGetLastError());
}

void vec_add2(float *z, const float *x, const float *y, int N, const int block_size) {
    const int grid_size = 16;
    vec_add_kernel2<<<grid_size, block_size>>>(z, x, y, N);
    cudaCheck(cudaGetLastError());
}

void vec_add(int kernel_num,
            float *z,
            const float *x, 
            const float *y,
            int N, int block_size
            ) {
    
    switch (kernel_num) {
        case 1:
            vec_add1(z, x, y, N, block_size);
            break;
        case 2:
            vec_add2(z, x, y, N, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }           
}

int main(int argc, const char **argv) {
    setup_main();

    int N = 1 << 20;
    float* x = make_random_float(N);
    float* y = make_random_float(N);
    float* z = (float*)malloc(N * sizeof(float));

    // read the kernel number from the command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }

    printf("Using kernel %d\n", kernel_num);

    // compute reference CPU solution
    vec_add_cpu(z, x, y, N);

    // move data to the GPU
    float *d_x, *d_y, *d_z;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_y, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_z, N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        vec_add(kernel_num, d_z, d_x, d_y, N, block_size);

        float tol = 0.0f;
        validate_result(d_z, z, "z", N, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, vec_add, kernel_num, d_z, d_x, d_y, N, block_size);

        // estimate memory bandwidth achieved
        // for each output element, we do 2 reads and 1 writes, 4 bytes each
        long memory_ops = N * 3 * (int)sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(z);
    free(x);
    free(y);

    cudaCheck(cudaFree(d_z));
    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_y));
    return 0;
}