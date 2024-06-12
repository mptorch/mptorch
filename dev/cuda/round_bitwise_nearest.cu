/*
Kernels for trailing significand rounding with the round to nearest even rule.
It assumes data is stored using a binary32 payload.


Compile example:
nvcc -O3 round_bitwise_nearest.cu -o round_bitwise_nearest

version 1 uses more temporary variables and an explicit if for testing
how masking should be done in the tie breaking rule
./round_bitwise_nearest 1

version 2 is a more compact version of 1 that avoids some of the temporaries
and the explicit if
./round_bitwise_nearest 2

*/

#include <cuda_runtime.h>

#include "common.h"

// --------------------------------------------------------------------------------------
// I/O pairs to sanity check the CPU reference code
uint32_t test_inputs[] = {
    0b00010110101010110001101001010011,   // round down - positive
    0b10010110101011010001101001010011,   // round down - negative
    0b00010110101011010001101001010011,   // round up   - positive
    0b10100110101010010001101001010011,   // round up   - negative
    0b00100110101011000000000000000000,   // round tie  - up
    0b10100110111111000000000000000000,   // round tie  - up + update
    0b00100110110101000000000000000000    // round tie  - down
};
uint32_t test_outputs[] = {
    0b00010110101010000000000000000000,
    0b10010110101100000000000000000000,
    0b00010110101100000000000000000000,
    0b10100110101010000000000000000000,
    0b00100110101100000000000000000000,
    0b10100111000000000000000000000000,
    0b00100110110100000000000000000000
};

// ---------------------------------------------------------------------------------------
// CPU reference code
uint32_t round_bitwise_nearest_cpu_impl(uint32_t target, int man_bits) {
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t half_eps = 1 << (22 - man_bits);
    // tie breaking rule offset
    int offset = (down == half_eps);
    uint32_t add_r = target + half_eps;
    return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

void round_bitwise_nearest_cpu(float *o, float *i, int N, int man_bits) {
    for (int j = 0; j < N; ++j) {
        uint32_t ival, oval;
        ival = FLOAT_TO_BITS(i + j);
        oval = round_bitwise_nearest_cpu_impl(ival, man_bits);
        o[j] = BITS_TO_FLOAT(&oval);
    }
}

// ---------------------------------------------------------------------------------------
// GPU kernels

__device__ __forceinline__ 
uint32_t round_bitwise_nearest_impl1(uint32_t target, int man_bits) {
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  // tie breaking rule offset
  uint32_t mask;
  if (down == (1 << (22 - man_bits))) {
    mask = (1 << (24 - man_bits)) - 1;
  } else {
    mask = (1 << (23 - man_bits)) - 1;
  }
  uint32_t half_eps = 1 << (23 - man_bits - 1);
  uint32_t add_r = target + half_eps;
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

__device__ __forceinline__
uint32_t round_bitwise_nearest_impl2(uint32_t target, int man_bits) {
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t half_eps = 1 << (22 - man_bits);
    // tie breaking rule offset
    int offset = (down == half_eps);
    uint32_t add_r = target + half_eps;
    return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

__global__ void round_bitwise_nearest_kernel1(float *o, float *__restrict__ i, int N, int man_bits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        uint32_t ival, oval;
        ival = FLOAT_TO_BITS(i + index);
        oval = round_bitwise_nearest_impl1(ival, man_bits);
        o[index] = BITS_TO_FLOAT(&oval);
    }
}

__global__ void round_bitwise_nearest_kernel2(float *o, float *__restrict__ i, int N, int man_bits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        uint32_t ival, oval;
        ival = FLOAT_TO_BITS(i + index);
        oval = round_bitwise_nearest_impl2(ival, man_bits);
        o[index] = BITS_TO_FLOAT(&oval);
    }
}

// ---------------------------------------------------------------------------------------
// Kernel launchers

void round_bitwise_nearest1(float *o, float *i, int N, int man_bits, const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    round_bitwise_nearest_kernel1<<<grid_size, block_size>>>(o, i, N, man_bits);
    cudaCheck(cudaGetLastError());
}

void round_bitwise_nearest2(float *o, float *i, int N, int man_bits, const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    round_bitwise_nearest_kernel2<<<grid_size, block_size>>>(o, i, N, man_bits);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void round_bitwise_nearest(int kernel_num, 
    float *o, 
    float *i, 
    int N, 
    int man_bits, 
    const int block_size) {

        switch (kernel_num) {
            case 1:
                round_bitwise_nearest1(o, i, N, man_bits, block_size);
                break;
            case 2:
                round_bitwise_nearest2(o, i, N, man_bits, block_size);
                break;
            default:
                printf("Invalid kernel number\n");
                exit(1);
        }

}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {

    setup_main();

    int N = 1 << 24;
    int man_bits = 4;

    // read the kernel number from the command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }

    // sanity check the CPU reference code
    for (int j = 0; j < sizeof(test_inputs) / sizeof(uint32_t); ++j) {
        uint32_t res = round_bitwise_nearest_cpu_impl(test_inputs[j], man_bits);
        if (res != test_outputs[j]) {
            printf("index = %d\n", j);
            print_uint32(res); printf("\nvs\n");
            print_uint32(test_outputs[j]); printf("\n");
            exit(EXIT_FAILURE);
        }
    }

    float* x = make_random_float(N);
    float* y = (float*)malloc(N * sizeof(float));

    printf("Using kernel %d\n", kernel_num);

    // compute reference CPU solution
    round_bitwise_nearest_cpu(y, x, N, man_bits);

    // move data to the GPU
    float *d_x, *d_y;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_y, N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        round_bitwise_nearest(kernel_num, d_y, d_x, N, man_bits, block_size);

        float tol = 0.0f;
        validate_result(d_y, y, "y", N, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, round_bitwise_nearest, kernel_num, d_y, d_x, N, man_bits, block_size);

        // estimate memory bandwidth achieved
        // for each output element, we do 1 read and 1 write, 4 bytes each
        long memory_ops = N * 2 * (int)sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(x);
    free(y);

    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_y));
    return 0;

}