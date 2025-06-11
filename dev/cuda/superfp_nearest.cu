/*
Kernels for IEEE-754 down casting from binary32 to a lower precision superfloat format.
Payload is still a binary32 value.

Compile example:
nvcc -O3 superfp_nearest.cu -o superfp_nearest -std=c++17 -lcublas

version 1 attempted to make the code as compact as possible, while also
maintaining readability; bit shifts and masking are used aplenty
./fp_nearest 1

*/

#include <cuda_runtime.h>

#include "common.h"

// --------------------------------------------------------------------------------------
// I/O pairs to sanity check the CPU reference code (superP3 - 1 binade, RNE)
uint32_t input_binade1[] = {
    0b11000000010100000000000000000000, // normalized range value - tie case (round to zero)
    0b01000000011100000000000000000000, // normalized range value - tie case (round away from zero)
    0b10111000000011001100110011001101, // normalized range value - round to zero case
    0b11011000100000000000000000000000, // overflow  range value - round to inf
    0b10110110100011001100110011001101, // subnormal range value - round to zero
    0b00110110111100110011001100110011, // subnormal range value - round away from zero
    0b11000111100110011001100110011010, // supnormal range value - round to zero
    0b11000111110000000000000000000000, // supnormal range value - tie case (round away from zero)
    0b11001000000000000000000000000000, // supnormal range value - exact
    0b01111111100000000000000000000000  // infinity
};

uint32_t output_binade1[] = {
    0b11000000010000000000000000000000,
    0b01000000100000000000000000000000,
    0b10111000000000000000000000000000,
    0b11111111100000000000000000000000,
    0b10110110100000000000000000000000,
    0b00110111000000000000000000000000,
    0b11000111100000000000000000000000,
    0b11001000000000000000000000000000,
    0b11001000000000000000000000000000,
    0b01111111100000000000000000000000};

// I/O pairs to sanity check the CPU reference code (superP3 - 2 binades, RNE)
uint32_t input_binade2[] = {
    0b11000000010100000000000000000000, // normalized range value - tie case (round to zero)
    0b01000000011100000000000000000000, // normalized range value - tie case (round away from zero)
    0b01000000100010001111010111000011, // normalized range value - round to zero case
    0b11011000100000000000000000000000, // overflow  range value - round to inf
    0b10110101001101000111101011100001, // subnormal range value - round to zero
    0b00110101110000010100011110101110, // subnormal range value - round away from zero
    0b11000111101000000010000011000101, // supnormal range value - round to zero
    0b11000111110000000000000000000000, // supnormal range value - tie case (round away from zero)
    0b11001000000000000000000000000000, // supnormal range value - exact
    0b01111111100000000000000000000000  // infinity
};

uint32_t output_binade2[] = {
    0b11000000010000000000000000000000,
    0b01000000100000000000000000000000,
    0b01000000100000000000000000000000,
    0b11111111100000000000000000000000,
    0b10110101000000000000000000000000,
    0b00110110000000000000000000000000,
    0b11000111100000000000000000000000,
    0b11001000000000000000000000000000,
    0b11001000000000000000000000000000,
    0b01111111100000000000000000000000};

// ---------------------------------------------------------------------------------------
// CPU reference code
uint32_t round_bitwise_nearest_cpu(uint32_t target, int man_bits)
{
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 1 << (22 - man_bits);
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    // apply the mask
    // this is the analogue of how you would do round
    // to nearest integer using the floor function:
    // round(x) = floor(x + 0.5)
    return add_r & ~((1 << (23 - man_bits + offset)) - 1);
};

// Remark: bias = 2^{e-1}
float cast_superfp_nearest_cpu(float origin, int man_bits, int exp_bits, int binades, bool saturate = false)
{
    int32_t sat = saturate;
    uint32_t target;
    target = FLOAT_TO_BITS(&origin);
    float ftarget{0u};

    int32_t target_exp = (target << 1 >> 24) - 127;
    int32_t min_exp = 1 - ((1 << (exp_bits - 1))) + (binades - 1);
    int32_t max_exp = ((1 << (exp_bits - 1)) - 2) - (binades - 1);
    bool subnormal = (target_exp < min_exp);
    bool supnormal = (target_exp > max_exp);
    if (subnormal)
    {
        if (target_exp < min_exp - binades * (1 << man_bits) + 1) // underflow
            return 0.0f;
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = BITS_TO_FLOAT(&qtarget);
    }
    else if (supnormal)
    {
        if (target_exp == 128)
        { // NaN/inf
            if (saturate)
            {
                if ((target & 0x7FFFFFFF) == 0x7F800000)
                { // inf
                    uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                    return BITS_TO_FLOAT(&qtarget);
                }
                else
                { // NaN
                    return origin;
                }
            }
            else
            {
                return origin;
            }
        }
        else if (target_exp >= max_exp + binades * (1 << man_bits) - 1 + sat)
        { // overflow
            if (saturate)
            {
                uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                return BITS_TO_FLOAT(&qtarget);
            }
            else
            {
                if (((target << 9) == 0u) && (target_exp == max_exp + binades * (1 << man_bits) - 1))
                    return origin;
                else
                {
                    float infty = INFINITY;
                    uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
                    return BITS_TO_FLOAT(&qtarget);
                }
            }
        }
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = BITS_TO_FLOAT(&qtarget);
    }
    else
    {
        uint32_t qtarget = round_bitwise_nearest_cpu(target, man_bits);
        ftarget = BITS_TO_FLOAT(&qtarget);
    }

    return ftarget;
}

void superfp_nearest_cpu(float *o, float *a, int N, int man_bits, int exp_bits, int binades, bool saturate)
{
    for (int i = 0; i < N; ++i)
        o[i] = cast_superfp_nearest_cpu(a[i], man_bits, exp_bits, binades, saturate);
}

// ---------------------------------------------------------------------------------------
// GPU kernels
__device__ __forceinline__ uint32_t round_bitwise_nearest_impl(uint32_t target, int man_bits)
{
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 1 << (22 - man_bits);
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    // apply the mask
    // this is the analogue of how you would do round
    // to nearest integer using the floor function:
    // round(x) = floor(x + 0.5)
    return add_r & ~((1 << (23 - man_bits + offset)) - 1);
};

// Remark: bias = 2^{e-1}
__device__ float cast_superfp_nearest_impl1(float origin, int man_bits, int exp_bits, int binades = 1, bool saturate = false)
{
    int32_t sat = saturate;
    uint32_t target;
    target = FLOAT_TO_BITS(&origin);
    float ftarget{0u};

    int32_t target_exp = (target << 1 >> 24) - 127;
    int32_t min_exp = 1 - ((1 << (exp_bits - 1))) + (binades - 1);
    int32_t max_exp = ((1 << (exp_bits - 1)) - 2) - (binades - 1);
    bool subnormal = (target_exp < min_exp);
    bool supnormal = (target_exp > max_exp);
    if (subnormal)
    {
        if (target_exp < min_exp - binades * (1 << man_bits) + 1) // underflow
            return 0.0f;
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = BITS_TO_FLOAT(&qtarget);
    }
    else if (supnormal)
    {
        if (target_exp == 128)
        { // NaN/inf
            if (saturate)
            {
                if ((target & 0x7FFFFFFF) == 0x7F800000)
                { // inf
                    uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                    return BITS_TO_FLOAT(&qtarget);
                }
                else
                { // NaN
                    return origin;
                }
            }
            else
            {
                return origin;
            }
        }
        else if (target_exp >= max_exp + binades * (1 << man_bits) - 1 + sat)
        { // overflow
            if (saturate)
            {
                uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                return BITS_TO_FLOAT(&qtarget);
            }
            else
            {
                if (((target << 9) == 0u) && (target_exp == max_exp + binades * (1 << man_bits) - 1))
                    return origin;
                else
                {
                    float infty = INFINITY;
                    uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
                    return BITS_TO_FLOAT(&qtarget);
                }
            }
        }
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = BITS_TO_FLOAT(&qtarget);
    }
    else
    {
        uint32_t qtarget = round_bitwise_nearest_impl(target, man_bits);
        ftarget = BITS_TO_FLOAT(&qtarget);
    }

    return ftarget;
};

__global__ void superfp_nearest_kernel1(float *o, float *__restrict__ a, int N,
                                        int man_bits, int exp_bits, int binades, bool saturate)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        o[index] = cast_superfp_nearest_impl1(a[index], man_bits, exp_bits, binades, saturate);
    }
}

// ---------------------------------------------------------------------------------------
// Kernel launchers
void superfp_nearest1(float *o, float *a, int N, int man_bits, int exp_bits,
                      int binades, bool saturate, const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    superfp_nearest_kernel1<<<grid_size, block_size>>>(o, a, N, man_bits, exp_bits, binades, saturate);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void superfp_nearest(int kernel_num,
                     float *o,
                     float *a,
                     int N,
                     int man_bits,
                     int exp_bits,
                     int binades,
                     bool saturate,
                     const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        superfp_nearest1(o, a, N, man_bits, exp_bits, binades, saturate, block_size);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv)
{

    setup_main();

    // read the kernel number from the command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }

    // sanity check the CPU reference code (1 binade case: "super"floats)
    for (int j = 0; j < sizeof(input_binade1) / sizeof(uint32_t); ++j)
    {
        float fres = cast_superfp_nearest_cpu(BITS_TO_FLOAT(&input_binade1[j]), 2, 5, 1, false);
        uint32_t res = FLOAT_TO_BITS(&fres);
        if (res != output_binade1[j])
        {
            printf("binade 1 test case:\n");
            printf("index = %d\n", j);
            print_float(res);
            printf("\nvs\n");
            print_uint32(output_binade1[j]);
            printf("\n");
            exit(EXIT_FAILURE);
        }
    }

    // sanity check the CPU reference code (2 binade case: "supreme"floats)
    for (int j = 0; j < sizeof(input_binade2) / sizeof(uint32_t); ++j)
    {
        float fres = cast_superfp_nearest_cpu(BITS_TO_FLOAT(&input_binade2[j]), 2, 5, 2, false);
        uint32_t res = FLOAT_TO_BITS(&fres);
        if (res != output_binade2[j])
        {
            printf("binade 2 test case:\n");
            printf("index = %d\n", j);
            print_float(res);
            printf("\nvs\n");
            print_uint32(output_binade2[j]);
            printf("\n");
            exit(EXIT_FAILURE);
        }
    }

    int N = 1 << 24;
    int man_bits = 2;
    int exp_bits = 5;
    int binades = 1;
    bool saturate = false;

    float *x = make_random_float(N);
    float *y = (float *)malloc(N * sizeof(float));

    printf("Using kernel %d\n", kernel_num);

    // compute reference CPU solution
    superfp_nearest_cpu(y, x, N, man_bits, exp_bits, binades, saturate);

    // move data to the GPU
    float *d_x, *d_y;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_y, N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        superfp_nearest(kernel_num, d_y, d_x, N, man_bits, exp_bits, binades, saturate, block_size);

        float tol = 0.0f;
        validate_result(d_y, y, "y", N, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_gpu_kernel(repeat_times, superfp_nearest,
                                                  kernel_num, d_y, d_x, N, man_bits, exp_bits,
                                                  binades, saturate, block_size);

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