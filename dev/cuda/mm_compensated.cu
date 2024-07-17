/*
Kernels for IEEE-754 down casting from binary32 to a lower precision format.
Payload is still a binary32 value.

Compile example:
nvcc -O3 mm_compensated.cu -o mm_compensated -std=c++17 -lcublas

*/


#include <cuda_runtime.h>
#include "common.h"
#include <random>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <tuple>

__host__ __device__ __forceinline__ uint32_t
round_bitwise_nearest(uint32_t target, int man_bits) {
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 1 << (22 - man_bits);
  // tie breaking rule offset
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  return add_r & ~((1 << std::min((23 - man_bits + offset),23)) - 1);
}

__device__ __forceinline__ uint32_t clip_exponent_without_subnormals(int exp_bits, int man_bits, uint32_t old_num,
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

__device__ __forceinline__ uint32_t clip_exponent_with_subnormals(int exp_bits, int man_bits, uint32_t old_num,
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

__device__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
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

template <size_t SHMEM_SIZE>
__global__ void mm_fp_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                                   float *__restrict__ c, int M, int K, int N,
                                   int man_add, int exp_add, int man_mul,
                                   int exp_mul, bool subnormals,
                                   bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0f;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_fp_nearest(tmp + cast_fp_nearest(s_a[ty * blockDim.x + j] *
                                                  s_b[j * blockDim.x + tx],
                                                  man_mul, exp_mul, subnormals,
                                                  saturate),
                            man_add, exp_add, subnormals, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = tmp;
}

template <size_t SHMEM_SIZE>
__global__ void mm_fp_nearest_compensated_impl(float *__restrict__ a, float *__restrict__ b,
                                   float *__restrict__ c, int M, int K, int N,
                                   int man_add, int exp_add, int man_mul,
                                   int exp_mul, bool subnormals,
                                   bool saturate) {


  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.0f;
  float compensated = 0.0f; // compensation variable for Kahan summation

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      float product = cast_fp_nearest(s_a[ty * blockDim.x + j] *
                                      s_b[j * blockDim.x + tx],
                                      man_mul, exp_mul, subnormals, saturate);

      float y = cast_fp_nearest(product - compensated, man_add, exp_add, subnormals, saturate);
      float t = cast_fp_nearest(sum + y, man_add, exp_add, subnormals, saturate);
      compensated = cast_fp_nearest((t - sum) - y, man_add, exp_add, subnormals, saturate);
      sum = t;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = sum;
}

void mm_fp_nearest(float *a, float *b, float *c, int M, int K, int N,
                   int man_add, int exp_add, int man_mul, int exp_mul,
                   bool subnormals, bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_fp_nearest_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, man_add, exp_add, man_mul,
                                  exp_mul, subnormals, saturate);
}

void mm_fp_nearest_compensated(float *a, float *b, float *c, int M, int K, int N,
                   int man_add, int exp_add, int man_mul, int exp_mul,
                   bool subnormals, bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_fp_nearest_compensated_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, man_add, exp_add, man_mul,
                                  exp_mul, subnormals, saturate);
}


// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
    int M = 1000;
    int K = 1000;
    int N = 1000;

    float *a = (float*)malloc(M * K * sizeof(float));
    float *b = (float*)malloc(K * N * sizeof(float));
    float *c = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices a and b with some values (for simplicity, using random values)
    for (int i = 0; i < M * K; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_c, M * N * sizeof(float)));

    // Copy data to the GPU
    cudaCheck(cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    int man_add = 23, exp_add = 8, man_mul = 23, exp_mul = 8;
    bool subnormals = true, saturate = true;

    std::vector<float> times_standard(10);
    std::vector<float> times_compensated(10);

    float sum_compensate = 0;
    float sum_standard = 0;

    // Benchmark standard kernel
    for (int i = 0; i < 10; ++i) {
        times_standard[i] = benchmark_kernel(1000, mm_fp_nearest, d_a, d_b, d_c, M, K, N,
                                             man_add, exp_add, man_mul, exp_mul, subnormals, saturate);
        sum_standard+=times_standard[i];
        // std::cout << "Standard kernel time: " << times_standard[i] << " ms\n";
    }
    // Benchmark compensated kernel
    for (int i = 0; i < 10; ++i) {
        times_compensated[i] = benchmark_kernel(1000, mm_fp_nearest_compensated, d_a, d_b, d_c, M, K, N,
                                                man_add, exp_add, man_mul, exp_mul, subnormals, saturate);
        // std::cout << "Compensated kernel time: " << times_compensated[i] << " ms\n";
        sum_compensate+=times_compensated[i];
    }
    float avg_time_standard = sum_standard / times_standard.size();
    float avg_time_compensated = sum_compensate / times_compensated.size();

    std::cout << "Average Standard kernel time: " << avg_time_standard << " ms\n";
    std::cout << "Average Compensated kernel time: " << avg_time_compensated << " ms\n";

    // Free device memory
    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}