/*
Custom precision matrix-matrix multiply with compensated Kahan summation

Compile example:
nvcc -O3 compensated_matmul.cu -o compensated_matmul -lcublas --extended-lambda

Run with:
./compensated_matmul
*/

#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#include "common.h"

// ------------------------------------------------------------------------------------
// Quantization functions and wrappers

__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest(uint32_t target, int man_bits)
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

__host__ __device__ uint32_t clip_exponent_with_subnormals(int exp_bits, int man_bits, uint32_t old_num,
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

__host__ __device__ uint32_t clip_exponent_without_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                              uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

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
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
        quantized_num = offset * (min_exponent_store << 23);
        quantized_num |= old_sign;
    }
    return quantized_num;
}

__host__ __device__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
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
            quantize_bits = not_uflow * round_bitwise_nearest(target, exp_diff);
            quantize_bits =
                clip_exponent_with_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
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
            quantize_bits = round_bitwise_nearest(target, man_bits);
            quantize_bits =
                clip_exponent_without_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

// --------------------------------------------------------------------------------------------------
// CPU Kernel
template <class Qadd, class Qmul>
void mm_cpu_kernel1(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
    // naive version
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.f;
            float comp_term = 0.f;
            float update = 0.f;
            float y = 0.f;
            float t = 0.f;
            for (int k = 0; k < K; ++k)
            {
                update = quant_mul(a[i * K + k] * b[k * N + j]);
                y = quant_add(update - comp_term);
                t = quant_add(sum + y);
                comp_term = quant_add(quant_add(t - sum) - y);
                sum = t;
            }
            c[i * N + j] = sum;
        }
}

// --------------------------------------------------------------------------------------------------
// GPU Kernels
template <size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void mm_kahan_kernel1(float *__restrict__ a, float *__restrict__ b,
                                 float *__restrict__ c, int M, int K, int N,
                                 Qadd quant_add, Qmul quant_mul)
{
    // declare shared memory matrices for A and B
    __shared__ float s_a[SHMEM_SIZE];
    __shared__ float s_b[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    float comp_term = 0.0f;
    float update = 0.0f;
    float y = 0.0f;
    float t = 0.0f;

    // sweep tile across matrix
    for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
    {
        // load in elements for this tile
        s_a[ty * blockDim.x + tx] =
            (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
        s_b[ty * blockDim.x + tx] =
            (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

        // wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // do matrix multiplication on the small matrices
        for (int j = 0; j < blockDim.x; ++j)
        {
            update = quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx]);
            y = quant_add(update - comp_term);
            t = quant_add(sum + y);
            comp_term = quant_add(quant_add(t - sum) - y);
            sum = t;
        }

        // wait for all threads to finish using current tiles
        // before landing in new ones
        __syncthreads();
    }

    // write back results
    if (row < M && col < N)
        c[row * N + col] = sum;
}

// --------------------------------------------------------------------------------------------------
// Kernel Launchers

void mm_kahan_cpu1(float *a, float *b, float *c, int M, int K, int N,
                   int man_add, int exp_add, int man_mul, int exp_mul,
                   bool subnormals, bool saturate)
{
    mm_cpu_kernel1(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
}

void mm_kahan_cuda1(float *a, float *b, float *c, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    bool subnormals, bool saturate)
{
    constexpr size_t THREADS_X{8U};
    constexpr size_t THREADS_Y{8U};
    constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
    dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
    dim3 const block_dim{
        (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
        (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
    mm_kahan_kernel1<SHMEM_SIZE><<<block_dim, thread_dim>>>(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                                            { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                                            { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); });
}

int main(int argc, const char **argv)
{
    setup_main();

    int M = 100;
    int N = 100;
    int K = 100;
    float *a = make_random_float(M * K);
    float *b = make_random_float(K * N);
    float *c = make_zeros_float(M * N);

    // move data to the GPU
    float *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_c, M * N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    mm_kahan_cpu1(a, b, c, M, K, N, 7, 8, 7, 8, true, true);
    mm_kahan_cuda1(d_a, d_b, d_c, M, K, N, 7, 8, 7, 8, true, true);
    printf("Checking if kernels results match...\n");
    float tol = 1e-2f;
    validate_result(d_c, c, "c", M * N, tol);
    printf("All results match. Starting benchmarks...\n\n");

    printf("CPU benchmarking...\n");
    int repeat_times = 100;
    float elapsed_time1 = benchmark_cpu_kernel(repeat_times, mm_kahan_cpu1, a, b, c, M, K, N, 7, 8, 7, 8, true, true);
    printf("time mm_kanan_cpu1 %4f ms\n", elapsed_time1);

    printf("CUDA benchmarking...\n");
    repeat_times = 1000;
    elapsed_time1 = benchmark_gpu_kernel(repeat_times, mm_kahan_cuda1, d_a, d_b, d_c, M, K, N, 7, 8, 7, 8, true, true);
    printf("time mm_kahan_cuda1 %.4f ms\n", elapsed_time1);

    // free memory
    free(a);
    free(b);
    free(c);

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    return 0;
}