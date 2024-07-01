#include "bit_helper.cu"
#include "quant_kernel.h"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// TODO: support saturate logic
// Remark: bias = 2^{e-1}
__device__ float cast_superfp_nearest(float origin, int man_bits, int exp_bits, int binades = 1, bool saturate = false) {
    int32_t sat = saturate;
    uint32_t target;
    target = FLOAT_TO_BITS(&origin);
    float ftarget{0u};

    int32_t target_exp = (target << 1 >> 24) - 127;
    int32_t min_exp = 1 - ((1 << (exp_bits - 1))) + (binades - 1);
    int32_t max_exp = ((1 << (exp_bits - 1)) - 2) - (binades - 1);
    bool subnormal = (target_exp < min_exp);
    bool supnormal = (target_exp > max_exp);
    if(subnormal) {
        if (target_exp < min_exp - binades * (1 << man_bits) + 1) // underflow
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
                    uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                    return BITS_TO_FLOAT(&qtarget);
                } else { // NaN
                    return origin;
                }
            } else {
                return origin;
            }
        } else if (target_exp >= max_exp + binades * (1 << man_bits) - 1 + sat) { // overflow
            if (saturate) {
                uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                return BITS_TO_FLOAT(&qtarget);
            } else {
                if (((target << 9) == 0u) && (target_exp == max_exp + binades * (1 << man_bits) - 1))
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
                                        int man_bits, int exp_bits, int binades,
                                        bool saturate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
        o[index] = cast_superfp_nearest(a[index], man_bits, exp_bits, binades, saturate);
}

template <size_t SHMEM_SIZE>
__global__ void mm_superfp_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                                   float *__restrict__ c, int M, int K, int N,
                                   int man_add, int exp_add, int man_mul,
                                   int exp_mul, int binades_add, int binades_mul, bool saturate) {

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
      tmp = cast_superfp_nearest(tmp + cast_superfp_nearest(s_a[ty * blockDim.x + j] *
                                                      s_b[j * blockDim.x + tx],
                                                  man_mul, exp_mul, binades_mul, saturate),
                            man_add, exp_add, binades_add, saturate);
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
__global__ void
bmm_superfp_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                    float *__restrict__ c, int M, int K, int N, int man_add,
                    int exp_add, int man_mul, int exp_mul, int binades_add,
                    int binades_mul, bool saturate) {
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0f;

  // Determine the start index of the current batch in the 1D linearized arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_superfp_nearest(tmp + cast_superfp_nearest(s_a[ty * blockDim.x + j] *
                                                      s_b[j * blockDim.x + tx],
                                                  man_mul, exp_mul, binades_mul,
                                                  saturate),
                            man_add, exp_add, binades_add, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write the result back to global memory
  if (row < M && col < N) {
    c[batch_c + row * N + col] = tmp;
  }
}

template <size_t SHMEM_SIZE>
__global__ void
mm_superfp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                       float *__restrict__ c, int M, int K, int N, int man_fma,
                       int exp_fma, int binades_fma, bool saturate) {
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
      tmp = cast_superfp_nearest(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
          man_fma, exp_fma, binades_fma, saturate);
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
__global__ void
bmm_superfp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                        float *__restrict__ c, int M, int K, int N, int man_fma,
                        int exp_fma, int binades_fma, bool saturate) {
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0f;

  // Determine the start index of the current batch in the 1D linearized arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_superfp_nearest(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
          man_fma, exp_fma, binades_fma, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write the result back to global memory
  if (row < M && col < N) {
    c[batch_c + row * N + col] = tmp;
  }
}

void mm_superfp_nearest(float *a, float *b, float *c, int M, int K, int N,
                   int man_add, int exp_add, int man_mul, int exp_mul,
                   int binades_add, int binades_mul, bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_superfp_nearest_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, man_add, exp_add, man_mul,
                                  exp_mul, binades_add, binades_mul, saturate);
}

void bmm_superfp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    int binades_add, int binades_mul, bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  bmm_superfp_nearest_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, man_add, exp_add, man_mul,
                                  exp_mul, binades_add, binades_mul, saturate);
}

void mm_superfp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                       int man_fma, int exp_fma, int binades_fma, bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_superfp_fma_nearest_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, man_fma, exp_fma, binades_fma, saturate);
}

void bmm_superfp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                        int N, int man_fma, int exp_fma, int binades_fma, bool saturate) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  bmm_superfp_fma_nearest_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, man_fma, exp_fma, binades_fma, saturate);
}