#include "bit_helper.cu"
#include "quant_kernel.h"
#include "softmax_kernel.h"
#include "layernorm_kernel.h"
#include "mm_kernel.h"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// Remark: bias = 2^{e-1}
__device__ float cast_superfp_nearest(float origin, int man_bits, int exp_bits, int binades_l = 1, int binades_u = 1, bool saturate = false) {
    int32_t sat = saturate;
    uint32_t target;
    target = FLOAT_TO_BITS(&origin);
    float ftarget{0u};

    int32_t target_exp = (target << 1 >> 24) - 127;
    int32_t min_exp = 1 - ((1 << (exp_bits - 1))) + (binades_l - 1);
    int32_t max_exp = ((1 << (exp_bits - 1)) - 2) - (binades_u - 1);
    bool subnormal = (target_exp < min_exp);
    bool supnormal = (target_exp > max_exp);
    if(subnormal) {
        if (target_exp < min_exp - binades_l * (1 << man_bits) + 1) // underflow
            return 0.0f;
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = BITS_TO_FLOAT(&qtarget);
    } else if (supnormal) {
        if (target_exp == 128) { // NaN/inf
            if (saturate) {
                if ((target & 0x7FFFFFFF) == 0x7F800000) { // inf
                    uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades_u * (1 << man_bits) + 127u) << 23);
                    return BITS_TO_FLOAT(&qtarget);
                } else { // NaN
                    return origin;
                }
            } else {
                return origin;
            }
        } else if (target_exp >= max_exp + binades_u * (1 << man_bits) - 1 + sat) { // overflow
            if (saturate) {
                uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades_u * (1 << man_bits) + 127u) << 23);
                return BITS_TO_FLOAT(&qtarget);
            } else {
                if (((target << 9) == 0u) && (target_exp == max_exp + binades_u * (1 << man_bits) - 1))
                    return origin;
                else {
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
    } else {
        uint32_t qtarget = round_bitwise_nearest(target, man_bits);
        ftarget = BITS_TO_FLOAT(&qtarget);
    }

    return ftarget;
}

__global__ void superfp_kernel_nearest(float *__restrict__ a, float *o, int size, 
                                        int man_bits, int exp_bits, 
                                        int binades_l, int binades_u,
                                        bool saturate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
        o[index] = cast_superfp_nearest(a[index], man_bits, exp_bits, binades_l, binades_u, saturate);
}

void mm_superfp_nearest(float *a, float *b, float *c, int M, int K, int N,
                   int man_add, int exp_add, int man_mul, int exp_mul,
                   int binades_add_l, int binades_add_u, 
                   int binades_mul_l, int binades_mul_u,
                   bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_impl<1u, SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, 
      [man_add, exp_add, binades_add_l, binades_add_u, saturate] __device__ (float x) { return cast_superfp_nearest(x, man_add, exp_add, binades_add_l, binades_add_u, saturate); },
      [man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate] __device__ (float x) { return cast_superfp_nearest(x, man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate); } 
  );
}

void bmm_superfp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    int binades_add_l, int binades_add_u,
                    int binades_mul_l, int binades_mul_u,
                    bool saturate) {

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
      [man_add, exp_add, binades_add_l, binades_add_u, saturate] __device__ (float x) { return cast_superfp_nearest(x, man_add, exp_add, binades_add_l, binades_add_u, saturate); },
      [man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate] __device__ (float x) { return cast_superfp_nearest(x, man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate); } 
  );
}

void mm_superfp_fma_nearest(float *a, float *b, float *c, 
                      int M, int K, int N,
                      int man_fma, int exp_fma, 
                      int binades_fma_l, int binades_fma_u,
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
      [man_fma, exp_fma, binades_fma_l, binades_fma_u, saturate] __device__ (float x) { return cast_superfp_nearest(x, man_fma, exp_fma, binades_fma_l, binades_fma_u, saturate); }
  );
}

void bmm_superfp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                        int N, int man_fma, int exp_fma, 
                        int binades_fma_l, int binades_fma_u, bool saturate) {

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
      [man_fma, exp_fma, binades_fma_l, binades_fma_u, saturate] __device__ (float x) { return cast_superfp_nearest(x, man_fma, exp_fma, binades_fma_l, binades_fma_u, saturate); }
  );
}


void softmax_forward_superfp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_exp, int exp_exp, int binades_exp_l, int binades_exp_u,
                                int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                bool saturate)
{
  softmax_forward(a, o, sizes,
    [man_exp, exp_exp, binades_exp_l, binades_exp_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_exp, exp_exp, binades_exp_l, binades_exp_u, saturate);
    },
    [man_off, exp_off, binades_off_l, binades_off_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_off, exp_off, binades_off_l, binades_off_u, saturate);
    },
    [man_acc, exp_acc, binades_acc_l, binades_acc_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_acc, exp_acc, binades_acc_l, binades_acc_u, saturate);
    }
  );
}

void softmax_lse_forward_superfp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                int man_lse, int exp_lse, int binades_lse_l, int binades_lse_u,
                                bool saturate)
{
  softmax_lse_forward(a, o, sizes,
    [man_off, exp_off, binades_off_l, binades_off_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_off, exp_off, binades_off_l, binades_off_u, saturate);
    },
    [man_lse, exp_lse, binades_lse_l, binades_lse_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_lse, exp_lse, binades_lse_l, binades_lse_u, saturate);
    }
  );
}

void softmax_backward_superfp_nearest(float *a, float *g, float *o,
                                const DimSizes& sizes,
                                int man_add, int exp_add, int binades_add_l, int binades_add_u,
                                int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                bool saturate)
{
  softmax_backward(a, g, o, sizes,
    [man_add, exp_add, binades_add_l, binades_add_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_add, exp_add, binades_add_l, binades_add_u, saturate);
    },
    [man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate);
    }
  );
}

void layernorm_forward_superfp_nearest(float *input, float *weight, float *bias,
                                      float *output, float *mean, float *rstd,
                                      float eps, const DimSizes& sizes,
                                      int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                      int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u, 
                                      int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                      int man_sqrt, int exp_sqrt, int binades_sqrt_l, int binades_sqrt_u,
                                      bool saturate)
{
  layernorm_forward(input, weight, bias, output, mean, rstd, eps, sizes,
    [man_acc, exp_acc, binades_acc_l, binades_acc_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_acc, exp_acc, binades_acc_l, binades_acc_u, saturate);
    },
    [man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate);
    },
    [man_div, exp_div, binades_div_l, binades_div_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_div, exp_div, binades_div_l, binades_div_u, saturate);
    },
    [man_sqrt, exp_sqrt, binades_sqrt_l, binades_sqrt_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_sqrt, exp_sqrt, binades_sqrt_l, binades_sqrt_u, saturate);
    }
  );
}

void layernorm_backward_superfp_nearest(float *input, float *grad_output,
                                      float *weight, float *bias,
                                      float *mean, float *rstd,
                                      float *grad_input, float *grad_gamma, float *grad_beta,
                                      const DimSizes& sizes,
                                      int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                      int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                      int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                      bool saturate)
{
  // creating xhat_gradient, an array of all 0s for backward pass
  // xhat_gradient is an output from the first pass of the backward
  // used again as an input to the second pass of the backward
  float* xhat_gradient;
  cudaMalloc(&xhat_gradient, sizeof(float) * sizes.outer * sizes.inner * sizes.channel);
  layernorm_backward(input, grad_output, weight, bias, mean, rstd, grad_input, grad_gamma, grad_beta, xhat_gradient, sizes,
    [man_acc, exp_acc, binades_acc_l, binades_acc_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_acc, exp_acc, binades_acc_l, binades_acc_u, saturate);
    },
    [man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_mul, exp_mul, binades_mul_l, binades_mul_u, saturate);
    },
    [man_div, exp_div, binades_div_l, binades_div_u, saturate] __device__ (float x) { 
      return cast_superfp_nearest(x, man_div, exp_div, binades_div_l, binades_div_u, saturate);
    }
  );
  cudaFree(xhat_gradient);
}