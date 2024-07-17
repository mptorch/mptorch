#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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
  // TODO
  return 0.0f;
}


__global__ void seed_init(uint32_t seed, curandState_t *state) {
  curand_init(seed, blockIdx.x * blockIdx.y, 0,
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
__global__ void
bmm_fp_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                    float *__restrict__ c, int M, int K, int N, int man_add,
                    int exp_add, int man_mul, int exp_mul, bool subnormals,
                    bool saturate) {
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

  // write the result back to global memory
  if (row < M && col < N) {
    c[batch_c + row * N + col] = tmp;
  }
}

template <size_t SHMEM_SIZE>
__global__ void
mm_fp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                       float *__restrict__ c, int M, int K, int N, int man_fma,
                       int exp_fma, bool subnormal_support, bool saturate) {
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
      tmp = cast_fp_nearest(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
          man_fma, exp_fma, subnormal_support, saturate);
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
bmm_fp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                        float *__restrict__ c, int M, int K, int N, int man_fma,
                        int exp_fma, bool subnormal_support, bool saturate) {
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
      tmp = cast_fp_nearest(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
          man_fma, exp_fma, subnormal_support, saturate);
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
__global__ void mm_fp_stochastic_impl(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    curandState_t *state, // int *__restrict__ r,
    int M, int K, int N, int man_add, int exp_add, int man_mul, int exp_mul,
    bool subnormal_support, bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x) * 2;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  uint32_t radd, rmul;

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
      radd = curand(&state[sidx]);
      rmul = curand(&state[sidx]);
      // radd = (uint32_t)r[bidx + 2 * (i + j)];
      // rmul = (uint32_t)r[bidx + 2 * (i + j) + 1];
      tmp = cast_fp_stochastic(
          tmp + cast_fp_stochastic(
                    s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], rmul,
                    man_mul, exp_mul, subnormal_support, saturate),
          radd, man_add, exp_add, subnormal_support, saturate);
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
__global__ void bmm_fp_stochastic_impl(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    curandState_t *state, // int *__restrict__ r,
    int M, int K, int N, int man_add, int exp_add, int man_mul, int exp_mul,
    bool subnormal_support, bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x) * 2;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  uint32_t radd, rmul;
  // Determine the start index of the current batch in the 1D linearized
  // arrays
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
      radd = curand(&state[sidx]);
      rmul = curand(&state[sidx]);
      // radd = (uint32_t)r[bidx + 2 * (i + j)];
      // rmul = (uint32_t)r[bidx + 2 * (i + j) + 1];
      tmp = cast_fp_stochastic(
          tmp + cast_fp_stochastic(
                    s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], rmul,
                    man_mul, exp_mul, subnormal_support, saturate),
          radd, man_add, exp_add, subnormal_support, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[batch_c + row * N + col] = tmp;
}

template <size_t SHMEM_SIZE>
__global__ void
mm_fp_fma_stochastic_impl(float *__restrict__ a, float *__restrict__ b,
                          float *__restrict__ c,
                          curandState_t *state, // int *__restrict__ r,
                          int M, int K, int N, int man_fma, int exp_fma,
                          bool subnormal_support, bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x);
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  uint32_t rfma;

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
      // rfma = (uint32_t)r[bidx + i + j];
      rfma = curand(&state[sidx]);
      tmp = cast_fp_stochastic(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp), rfma,
          man_fma, exp_fma, subnormal_support, saturate);
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
bmm_fp_fma_stochastic_impl(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c,
                           curandState_t *state, // int *__restrict__ r,
                           int M, int K, int N, int man_fma, int exp_fma,
                           bool subnormal_support, bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x);
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  uint32_t rfma;

  // Determine the start index of the current batch in the 1D linearized
  // arrays
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
      // rfma = (uint32_t)r[bidx + i + j];
      rfma = curand(&state[sidx]);
      tmp = cast_fp_stochastic(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp), rfma,
          man_fma, exp_fma, subnormal_support, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[batch_c + row * N + col] = tmp;
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
  bmm_fp_nearest_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, man_add, exp_add, man_mul,
                                  exp_mul, subnormals, saturate);
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
  mm_fp_fma_nearest_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, man_fma, exp_fma, subnormals, saturate);
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
  bmm_fp_fma_nearest_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, man_fma, exp_fma, subnormals, saturate);
}

void mm_fp_stochastic(float *a, float *b, float *c, int M, int K, int N,
                      int man_add, int exp_add, int man_mul, int exp_mul,
                      bool subnormal_support, bool saturate) {
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
  seed_init<<<block_dim, 1>>>(time(0), state);
  mm_fp_stochastic_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c,
      state, // rand_ints.data<int>(),
      M, K, N, man_add, exp_add, man_mul, exp_mul, subnormal_support, saturate);
  cudaFree(state);
}

void bmm_fp_stochastic(float *a, float *b, float *c, int B, int M, int K, int N,
                       int man_add, int exp_add, int man_mul, int exp_mul,
                       bool subnormal_support, bool saturate) {
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
  seed_init<<<block_dim, 1>>>(time(0), state);
  bmm_fp_stochastic_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c,
      state, // rand_ints.data<int>(),
      M, K, N, man_add, exp_add, man_mul, exp_mul, subnormal_support, saturate);
  cudaFree(state);
}

void mm_fp_fma_stochastic(float *a, float *b, float *c, int M, int K, int N,
                          int man_fma, int exp_fma, bool subnormal_support,
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
  seed_init<<<block_dim, 1>>>(time(0), state);
  mm_fp_fma_stochastic_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c,
      state, // rand_ints.data_ptr<int>(),
      M, K, N, man_fma, exp_fma, subnormal_support, saturate);
  cudaFree(state);
}

void bmm_fp_fma_stochastic(float *a, float *b, float *c, int B, int M, int K,
                           int N, int man_fma, int exp_fma,
                           bool subnormal_support, bool saturate) {
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
  seed_init<<<block_dim, 1>>>(time(0), state);
  bmm_fp_fma_stochastic_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c,
      state, // rand_ints.data_ptr<int>(),
      M, K, N, man_fma, exp_fma, subnormal_support, saturate);
  cudaFree(state);
}


template<class Qexp, class Qoff, class Qacc, class Qdiv>
__global__ void softmax_forward_impl(
    const float* __restrict__ input_array, float *output_array, const DimStrides *strides,
    Qexp quant_exp, Qoff quant_off, Qacc quant_acc, Qdiv quant_div)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / strides->inner_size;
    int inner_idx = blockIdx.x % strides->inner_size;
    
    int base_index = outer_idx * strides->outer_stride + inner_idx;
    const float* input = input_array + base_index;
    float* output = output_array + base_index;

    // compute the maximum value of the row
    
    float max = -INFINITY;
    // each thread computes part of the maximum by iterating over its corresponding
    // position along the row, as many times as required to cover all the row
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        max = fmaxf(max, input[idx]);
    }
    // intra-warp maximum reduction
    // each thread now contains part of the maximum of the row, we combine these maximums
    // into a per-warp maximum, stored in the 0th thread of the warp (lane = 0)
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, max, offset);
        max = fmaxf(max, other_max);
    }
    // store the warp-level maximum in shared memory
    if (lane == 0) {
        shared[warp] = max;
    }
    __syncthreads();
    // reduce the maximum of each warp into the 0th thread of the block (warp 0, lane 0)
    if(tid == 0) {
        max = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            max = fmaxf(max, shared[i]);
        }
        // store the block's maximum (i.e row's maximum) in a fixed shared location
        shared[0] = max;
    }
    __syncthreads();
    // each thread reads the row's maximum
    max = shared[0];

    // each thread computes its exp(x[i] - max)
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        output[idx] = quant_exp(expf(quant_off(input[idx] - max)));
    }

    // compute the sum of exp(x[i] - max), using a similar approach as for the maximum,
    // but reducing a sum instead

    float sum = 0.f;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        sum = quant_acc(sum + output[idx]); // we sum the previously computed exponentials
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum = quant_acc(sum + __shfl_down_sync(0xFFFFFFFF, sum, offset));
    }
    if (lane == 0) {
        shared[warp] = sum;
    }
    __syncthreads();
    if(tid == 0) {
        sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            sum = quant_acc(sum + shared[i]);
        }
        shared[0] = sum;
    }
    __syncthreads();
    sum = shared[0];

    // finally, divide each previously computed exponentials by the sum
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        output[idx] = quant_div(output[idx] / sum);
    }
}

template<class Qexp, class Qoff, class Qlse>
__global__ void softmax_lse_forward_impl(
    const float* __restrict__ input_array, float *output_array, const DimStrides *strides,
    Qexp quant_exp, Qoff quant_off, Qlse quant_lse)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / strides->inner_size;
    int inner_idx = blockIdx.x % strides->inner_size;
    
    int base_index = outer_idx * strides->outer_stride + inner_idx;
    const float* input = input_array + base_index;
    float* output = output_array + base_index;

    // compute the maximum value of the row

    float max = -INFINITY;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        max = fmaxf(max, input[idx]);
    }
    // intra-warp maximum reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, max, offset);
        max = fmaxf(max, other_max);
    }
    // store the warp-level maximum in shared memory
    if (lane == 0) {
        shared[warp] = max;
    }
    __syncthreads();
    // reduce the maximum of each warp into the 0th thread of the block (warp 0, lane 0)
    if(tid == 0) {
        max = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            max = fmaxf(max, shared[i]);
        }
        // store the block's maximum (i.e row's maximum) in a fixed shared location
        shared[0] = max;
    }
    __syncthreads();
    // each thread reads the row's maximum
    max = shared[0];

    // compute log(exp(x[i] - max) + ...) using LogSumExp iterations
    float lgs = -INFINITY;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        float x = quant_off(input[idx] - max);
        output[idx] = x;
        lgs = quant_lse(logf(expf(lgs) + expf(x)));
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_lgs = __shfl_down_sync(0xFFFFFFFF, lgs, offset);
        lgs = quant_lse(logf(expf(lgs) + expf(other_lgs)));
    }
    if (lane == 0) {
        shared[warp] = lgs;
    }
    __syncthreads();
    if(tid == 0) {
        lgs = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            lgs = quant_lse(logf(expf(lgs) + expf(shared[i])));
        }
        shared[0] = lgs;
    }
    __syncthreads();
    lgs = shared[0];

    // finally, divide each previously computed exponentials by the sum
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        output[idx] = quant_exp(expf(quant_off(output[idx] - lgs)));
    }
}

template<class Qadd, class Qmul, class Qdiv>
__global__ void softmax_backward_impl(
  const float* __restrict__ input_array, const float* __restrict__ out_gradient, float* output_array,
  const DimStrides *strides,
  Qadd quant_add, Qmul quant_mul, Qdiv quant_div)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / strides->inner_size;
    int inner_idx = blockIdx.x % strides->inner_size;
    
    int base_index = outer_idx * strides->outer_stride + inner_idx;
    const float* input = input_array + base_index;
    const float* grad = out_gradient + base_index;
    float* output = output_array + base_index;

    // Compute the input row sum and weighted sum

    float* shared_input_sum = &shared[0];
    float* shared_weighted_grad_sum = &shared[warpsPerBlock];

    float input_sum = 0.f;
    float weighted_grad_sum = 0.f;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        input_sum = quant_add(input_sum + input[idx]);
        float prod = quant_mul(input[idx] * grad[idx]);
        weighted_grad_sum = quant_add(weighted_grad_sum + prod);
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        input_sum = quant_add(input_sum + __shfl_down_sync(0xFFFFFFFF, input_sum, offset));
        weighted_grad_sum = quant_add(weighted_grad_sum + __shfl_down_sync(0xFFFFFFFF, weighted_grad_sum, offset));
    }
    if (lane == 0) {
        shared_input_sum[warp] = input_sum;
        shared_weighted_grad_sum[warp] = weighted_grad_sum;
    }
    __syncthreads();
    if(tid == 0) {
        input_sum = shared_input_sum[0];
        weighted_grad_sum = shared_weighted_grad_sum[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            input_sum = quant_add(input_sum + shared_input_sum[i]);
            weighted_grad_sum = quant_add(weighted_grad_sum + shared_weighted_grad_sum[i]);
        }
        shared_input_sum[0] = input_sum;
        shared_weighted_grad_sum[0] = weighted_grad_sum;
    }
    __syncthreads();
    input_sum = shared_input_sum[0];
    weighted_grad_sum = shared_weighted_grad_sum[0];

    // Last step, subsrtact the weighted sum from the gradient, and divide by input sum
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        float a = quant_add(grad[idx] - weighted_grad_sum);
        float b = quant_mul(a * input[idx]);
        output[idx] = quant_div(b / input_sum);
    }
}


void softmax_forward_fp_nearest(float *a, float *o,
                                const DimStrides& strides,
                                int man_exp, int exp_exp,
                                int man_off, int exp_off,
                                int man_acc, int exp_acc,
                                int man_div, int exp_div,
                                bool subnormals, bool saturate)
{
  DimStrides *d_strides;
  cudaMalloc(&d_strides, sizeof(DimStrides));
  cudaMemcpy(d_strides, &strides, sizeof(DimStrides), cudaMemcpyHostToDevice);
  int blocks = strides.outer_size * strides.inner_size; 
  int block_size = 64;
  size_t shared_mem_size = block_size * sizeof(float);
  softmax_forward_impl<<<blocks, block_size, shared_mem_size>>>(
    a, o, d_strides,
    [man_exp, exp_exp, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_exp, exp_exp, subnormals, saturate); },
    [man_off, exp_off, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_off, exp_off, subnormals, saturate); },
    [man_acc, exp_acc, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_acc, exp_acc, subnormals, saturate); },
    [man_div, exp_div, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_div, exp_div, subnormals, saturate); }
  );
  cudaFree(d_strides);
}

void softmax_lse_forward_fp_nearest(float *a, float *o,
                                const DimStrides& strides,
                                int man_exp, int exp_exp,
                                int man_off, int exp_off,
                                int man_lse, int exp_lse,
                                bool subnormals, bool saturate)
{
  DimStrides *d_strides;
  cudaMalloc(&d_strides, sizeof(DimStrides));
  cudaMemcpy(d_strides, &strides, sizeof(DimStrides), cudaMemcpyHostToDevice);
  int blocks = strides.outer_size * strides.inner_size; 
  int block_size = 64;
  size_t shared_mem_size = block_size * sizeof(float);
  softmax_lse_forward_impl<<<blocks, block_size, shared_mem_size>>>(
    a, o, d_strides,
    [man_exp, exp_exp, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_exp, exp_exp, subnormals, saturate); },
    [man_off, exp_off, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_off, exp_off, subnormals, saturate); },
    [man_lse, exp_lse, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_lse, exp_lse, subnormals, saturate); }
  );
  cudaFree(d_strides);
}

void softmax_backward_fp_nearest(float *a, float *g, float *o,
                                const DimStrides& strides,
                                int man_add, int exp_add,
                                int man_mul, int exp_mul,
                                int man_div, int exp_div,
                                bool subnormals, bool saturate)
{
  DimStrides *d_strides;
  cudaMalloc(&d_strides, sizeof(DimStrides));
  cudaMemcpy(d_strides, &strides, sizeof(DimStrides), cudaMemcpyHostToDevice);
  int blocks = strides.outer_size * strides.inner_size; 
  int block_size = 64;
  size_t shared_mem_size = 2 * block_size * sizeof(float);
  softmax_backward_impl<<<blocks, block_size, shared_mem_size>>>(
    a, g, o, d_strides,
    [man_add, exp_add, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); },
    [man_mul, exp_mul, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); },
    [man_div, exp_div, subnormals, saturate] __device__ (float x) { return cast_fp_nearest(x, man_div, exp_div, subnormals, saturate); }
  );
  cudaFree(d_strides);
}