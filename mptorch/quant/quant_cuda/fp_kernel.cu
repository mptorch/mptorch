#include "bit_helper.cu"
#include "quant_kernel.h"
#include "mm_kernel.h"
#include "sim_helper.cu"
#include "layernorm_kernel.h"
#include "softmax_kernel.h"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
// #include "binary8_kernel.cu"
// #include "binary8_kernel.h"

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
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

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

__device__ float cast_fp_stochastic(float origin_float, uint32_t rand_prob,
                                    int man_bits, int exp_bits,
                                    bool subnormal_support = true,
                                    bool saturate = false) {
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);

  if (subnormal && subnormal_support) {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    shift_float = BITS_TO_FLOAT(&shift_bits);
    val = origin_float + shift_float;
    target = FLOAT_TO_BITS(&val);
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
  } else {
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantize_bits =
        clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
    quantized = BITS_TO_FLOAT(&quantize_bits);
  }

  return quantized;
}

__device__ float cast_fp_stochastic(float origin_float, uint32_t rand_prob, int rand_bits,
                                    int man_bits, int exp_bits,
                                    bool subnormal_support = true,
                                    bool saturate = false) 
{
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);

  rand_prob = rand_prob << 9 >> 9;
  rand_prob = rand_prob & ~(1 << (23 - man_bits - rand_bits) - 1);

  if (subnormal && subnormal_support) {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    shift_float = BITS_TO_FLOAT(&shift_bits);
    val = origin_float + shift_float;
    target = FLOAT_TO_BITS(&val);
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
  } else {
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantize_bits =
        clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
    quantized = BITS_TO_FLOAT(&quantize_bits);
  }

  return quantized;
}


__global__ void seed_init(curandState_t *state) {
  curand_init(clock64(), blockIdx.x * blockIdx.y, 0,
              &state[blockIdx.x * blockIdx.y]);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits,
                                        bool subnormal_support, bool saturate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_stochastic(a[index], (uint32_t)r[index], man_bits,
                                  exp_bits, subnormal_support, saturate);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     int man_bits, int exp_bits,
                                     bool subnormal_support, bool saturate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_nearest(a[index], man_bits, exp_bits, subnormal_support,
                               saturate);
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
  mm_impl<1u, SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N,
      [man_add, exp_add, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); },
      [man_mul, exp_mul, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); }
  );
}

void bmm_fp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    bool subnormals, bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  bmm_impl<1u, SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N,
      [man_add, exp_add, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); },
      [man_mul, exp_mul, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); }
  );
}

void mm_fp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                       int man_fma, int exp_fma, bool subnormals,
                       bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_fma_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, 
      [man_fma, exp_fma, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_fma, exp_fma, subnormals, saturate); }
  );
}

void bmm_fp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                        int N, int man_fma, int exp_fma, bool subnormals,
                        bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  bmm_fma_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, 
      [man_fma, exp_fma, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_fma, exp_fma, subnormals, saturate); }
  );
}

void mm_fp_stochastic(float *a, float *b, float *c, int M, int K, int N,
                      int man_add, int exp_add, int man_mul, int exp_mul,
                      bool subnormals, bool saturate) {
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  mm_sr_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state, 
      M, K, N, 
      [man_add, exp_add, subnormals, saturate] __device__ (float x, uint32_t rnd) { return cast_fp_stochastic(x, rnd, man_add, exp_add, subnormals, saturate); },
      [man_mul, exp_mul, subnormals, saturate] __device__ (float x, uint32_t rnd) { return cast_fp_stochastic(x, rnd, man_mul, exp_mul, subnormals, saturate); }
  );
  cudaFree(state);
}

void bmm_fp_stochastic(float *a, float *b, float *c, int B, int M, int K, int N,
                       int man_add, int exp_add, int man_mul, int exp_mul,
                       bool subnormals, bool saturate) {
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  bmm_sr_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state,
      M, K, N,
      [man_add, exp_add, subnormals, saturate] __device__ (float x, uint32_t rnd) { return cast_fp_stochastic(x, rnd, man_add, exp_add, subnormals, saturate); },
      [man_mul, exp_mul, subnormals, saturate] __device__ (float x, uint32_t rnd) { return cast_fp_stochastic(x, rnd, man_mul, exp_mul, subnormals, saturate); }
);
  cudaFree(state);
}

void mm_fp_fma_stochastic(float *a, float *b, float *c, int M, int K, int N,
                          int man_fma, int exp_fma, bool subnormals,
                          bool saturate) {
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  mm_sr_fma_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state, 
      M, K, N, 
      [man_fma, exp_fma, subnormals, saturate] __device__ (float x, uint32_t rnd) { return cast_fp_stochastic(x, rnd, man_fma, exp_fma, subnormals, saturate); }
  );
  cudaFree(state);
}

void bmm_fp_fma_stochastic(float *a, float *b, float *c, int B, int M, int K,
                           int N, int man_fma, int exp_fma,
                           bool subnormals, bool saturate) {
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  bmm_sr_fma_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state,
      M, K, N,
      [man_fma, exp_fma, subnormals, saturate] __device__ (float x, uint32_t rnd) { return cast_fp_stochastic(x, rnd, man_fma, exp_fma, subnormals, saturate); }
  );
  cudaFree(state);
}

void softmax_forward_fp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_exp, int exp_exp,
                                int man_off, int exp_off,
                                int man_acc, int exp_acc,
                                bool subnormals, bool saturate)
{
  softmax_forward(a, o, sizes,
    [man_exp, exp_exp, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_exp, exp_exp, subnormals, saturate);
    },
    [man_off, exp_off, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_off, exp_off, subnormals, saturate);
    },
    [man_acc, exp_acc, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_acc, exp_acc, subnormals, saturate);
    }
  );
}

void softmax_lse_forward_fp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_off, int exp_off,
                                int man_lse, int exp_lse,
                                bool subnormals, bool saturate)
{
  softmax_lse_forward(a, o, sizes,
    [man_off, exp_off, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_off, exp_off, subnormals, saturate);
    },
    [man_lse, exp_lse, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_lse, exp_lse, subnormals, saturate);
    }
  );
}

void softmax_backward_fp_nearest(float *a, float *g, float *o,
                                const DimSizes& sizes,
                                int man_add, int exp_add,
                                int man_mul, int exp_mul,
                                bool subnormals, bool saturate)
{
  softmax_backward(a, g, o, sizes,
    [man_add, exp_add, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate);
    },
    [man_mul, exp_mul, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate);
    }
  );
}

void layernorm_forward_fp_nearest(float *input, float *weight, float *bias,
                              float *output, float *mean, float *rstd,
                              float eps, const DimSizes& sizes,
                              int man_acc, int exp_acc,
                              int man_mul, int exp_mul,
                              int man_div, int exp_div,
                              int man_sqrt, int exp_sqrt,
                              bool subnormals, bool saturate)
{
  layernorm_forward(input, weight, bias, output, mean, rstd, eps, sizes,
    [man_acc, exp_acc, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_acc, exp_acc, subnormals, saturate);
    },
    [man_mul, exp_mul, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate);
    },
    [man_div, exp_div, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_div, exp_div, subnormals, saturate);
    },
    [man_sqrt, exp_sqrt, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_sqrt, exp_sqrt, subnormals, saturate);
    }
  );
}

void layernorm_backward_fp_nearest(float *input, float *grad_output,
                              float *weight, float *bias,
                              float *mean, float *rstd,
                              float *grad_input, float *grad_gamma, float *grad_beta,
                              const DimSizes& sizes,
                              int man_acc, int exp_acc,
                              int man_mul, int exp_mul,
                              int man_div, int exp_div,
                              bool subnormals, bool saturate)
{
  // creating xhat_gradient, an array of all 0s for backward pass
  // xhat_gradient is an output from the first pass of the backward
  // used again as an input to the second pass of the backward
  float* xhat_gradient;
  cudaMalloc(&xhat_gradient, sizeof(float) * sizes.outer * sizes.inner * sizes.channel);
  layernorm_backward(input, grad_output, weight, bias, mean, rstd, grad_input, grad_gamma, grad_beta, xhat_gradient, sizes,
    [man_acc, exp_acc, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_acc, exp_acc, subnormals, saturate);
    },
    [man_mul, exp_mul, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate);
    },
    [man_div, exp_div, subnormals, saturate] __device__ (float x) { 
      return cast_fp_nearest(x, man_div, exp_div, subnormals, saturate);
    }
  );
  cudaFree(xhat_gradient);
}