/*
Kernels for IEEE-754 down casting from binary32 to a lower precision format.
Payload is still a binary32 value.

Compile example:
nvcc -O3 fp_nearest.cu -o fp_nearest -std=c++17 -lcublas

version 1 attempted to make the code as compact as possible, while also
maintaining readability; bit shifts and masking are used aplenty
./fp_nearest 1

*/

#include <cuda_runtime.h>

#include "common.h"
#include "half.h"

// --------------------------------------------------------------------------------------
// I/O pairs to sanity check the CPU reference code (E5M2 without subnormals, RNE)
uint32_t test_inputs[] = {
    0b00111000100000000000000000000000,
    0b00110100011111111111111111111111,
    0b00111000010000000000000000000000,
    0b00111000011000000000000000000000,
    0b10111000001000000000000000000000};
uint32_t test_outputs[] = {
    0b00111000100000000000000000000000,
    0b00000000000000000000000000000000,
    0b00111000100000000000000000000000,
    0b00111000100000000000000000000000,
    0b10111000100000000000000000000000};

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
}

uint32_t clip_subnormal_range_exponent_cpu(int exp_bits, int man_bits, uint32_t old_num,
                                           uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) - man_bits + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (1u << 23);
        quantized_num = quantized_num | old_sign;
        quantized_num = offset * quantized_num;
    }
    return quantized_num;
}

uint32_t clip_normal_range_exponent_cpu(int exp_bits, int man_bits, uint32_t old_num,
                                        uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int old_exponent_store = old_num << 1 >> 24;
    int quantized_exponent_store = quantized_num << 1 >> 24;
    int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // saturate or overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        if (saturate)
        {
            uint32_t max_man =
                (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
            uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
            quantized_num = old_sign | max_num;
        }
        else
        {
            quantized_num = ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
            quantized_num = quantized_num | old_sign;
        }
    } // underflow or round to smallest nonzero normal value
    else if (old_exponent_store < min_exponent_store)
    {
        uint32_t offset = (old_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > 0);
        quantized_num = offset * (min_exponent_store << 23);
        quantized_num |= old_sign;
    }
    return quantized_num;
}

float cast_fp_nearest_cpu(float origin_float, int man_bits, int exp_bits,
                          bool subnormal_support = true,
                          bool saturate = false)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && subnormal_support)
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest_cpu(target, exp_diff);
            quantize_bits =
                clip_subnormal_range_exponent_cpu(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
        // handle NaN/inf inputs
        else if (target_exp == 128)
        {
            quantized = origin_float;
        }
        // normal value range or overflow
        else
        {
            quantize_bits = round_bitwise_nearest_cpu(target, man_bits);
            quantize_bits =
                clip_normal_range_exponent_cpu(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

void fp_nearest_cpu(float *o, float *a, int N, int man_bits, int exp_bits, bool subnormal_support, bool saturate)
{
    for (int i = 0; i < N; ++i)
        o[i] = cast_fp_nearest_cpu(a[i], man_bits, exp_bits, subnormal_support, saturate);
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
}

__device__ uint32_t clip_subnormal_range_exponent_impl1(int exp_bits, int man_bits, uint32_t old_num,
                                                        uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) - man_bits + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (1u << 23);
        quantized_num = quantized_num | old_sign;
        quantized_num = offset * quantized_num;
    }
    return quantized_num;
}

__device__ uint32_t clip_normal_range_exponent_impl1(int exp_bits, int man_bits, uint32_t old_num,
                                                     uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int old_exponent_store = old_num << 1 >> 24;
    int quantized_exponent_store = quantized_num << 1 >> 24;
    int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // saturate or overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        if (saturate)
        {
            uint32_t max_man =
                (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
            uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
            quantized_num = old_sign | max_num;
        }
        else
        {
            quantized_num = ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
            quantized_num = quantized_num | old_sign;
        }
    } // underflow or round to smallest nonzero normal value
    else if (old_exponent_store < min_exponent_store)
    {
        uint32_t offset = (old_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > 0);
        quantized_num = offset * (min_exponent_store << 23);
        quantized_num |= old_sign;
    }
    return quantized_num;
}

__device__ float cast_fp_nearest_impl1(float origin_float, int man_bits, int exp_bits,
                                       bool subnormal_support = true,
                                       bool saturate = false)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && subnormal_support)
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest_impl(target, exp_diff);
            quantize_bits =
                clip_subnormal_range_exponent_impl1(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
        // handle NaN/inf inputs
        else if (target_exp == 128)
        {
            quantized = origin_float;
        }
        // normal value range or overflow
        else
        {
            quantize_bits = round_bitwise_nearest_impl(target, man_bits);
            quantize_bits =
                clip_normal_range_exponent_impl1(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

__global__ void fp_nearest_kernel1(float *o, float *__restrict__ a,
                                   int N, int man_bits, int exp_bits,
                                   bool subnormal_support, bool saturate)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        o[index] = cast_fp_nearest_impl1(a[index], man_bits, exp_bits, subnormal_support, saturate);
    }
}

// ---------------------------------------------------------------------------------------
// Kernel launchers
void fp_nearest1(float *o, float *a, int N, int man_bits, int exp_bits,
                 bool subnormal_support, bool saturate, const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    fp_nearest_kernel1<<<grid_size, block_size>>>(o, a, N, man_bits, exp_bits, subnormal_support, saturate);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void fp_nearest(int kernel_num,
                float *o,
                float *a,
                int N,
                int man_bits,
                int exp_bits,
                bool subnormal_support,
                bool saturate,
                const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        fp_nearest1(o, a, N, man_bits, exp_bits, subnormal_support, saturate, block_size);
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

    // sanity check the CPU reference code (no subnormals)
    for (int j = 0; j < sizeof(test_inputs) / sizeof(uint32_t); ++j)
    {
        float fres = cast_fp_nearest_cpu(BITS_TO_FLOAT(&test_inputs[j]), 2, 5, false);
        uint32_t res = FLOAT_TO_BITS(&fres);
        if (res != test_outputs[j])
        {
            printf("index = %d\n", j);
            printf("input = ");
            print_uint32(test_inputs[j]);
            printf("\n");
            printf("Outputs:\n");
            print_uint32(res);
            printf("\nvs\n");
            print_uint32(test_outputs[j]);
            printf("\n");
            exit(EXIT_FAILURE);
        }
    }

    // compare CPU reference code against a IEEE-754 binary16 implementation (subnormals on)
    using half_float::half;

    uint32_t max_val = (1ul << 32) - 1;

    for (uint32_t v = 0u; v < max_val; ++v)
    {
        float fval = BITS_TO_FLOAT(&v);
        half hval = (half)fval;
        float fval1 = (float)hval;
        float fval2 = cast_fp_nearest_cpu(fval, 10, 5, true, false);
        if ((fval1 != fval2))
        {
            if (!isnanf(fval1) or !isnanf(fval2))
            {
                printf("%.4f %.4f %.4f\n", fval, fval1, fval2);
                print_float(fval);
                printf("\n");
                print_float(fval1);
                printf("\n");
                print_float(fval2);
                printf("\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    int N = 1 << 24;
    int man_bits = 2;
    int exp_bits = 5;
    bool subnormal_support = true;
    bool saturate = false;

    float *x = make_random_float(N);
    float *y = (float *)malloc(N * sizeof(float));

    printf("Using kernel %d\n", kernel_num);

    // compute reference CPU solution
    fp_nearest_cpu(y, x, N, man_bits, exp_bits, subnormal_support, saturate);

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
        fp_nearest(kernel_num, d_y, d_x, N, man_bits, exp_bits, subnormal_support, saturate, block_size);

        float tol = 0.0f;
        validate_result(d_y, y, "y", N, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_gpu_kernel(repeat_times, fp_nearest,
                                                  kernel_num, d_y, d_x, N, man_bits, exp_bits,
                                                  subnormal_support, saturate, block_size);

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