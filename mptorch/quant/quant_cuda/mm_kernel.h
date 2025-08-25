#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

template <class RandType>
__device__ __forceinline__ RandType gen_rand(curandState_t *state, int sidx)
{
  return curand(&state[sidx]);
}

template <>
__device__ __forceinline__ float gen_rand<float>(curandState_t *state, int sidx)
{
  return 1.0f - curand_uniform(&state[sidx]);
}

template <size_t BLOCK_FACTOR, size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void mm_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                        int M, int K, int N,
                        Qadd quant_add, Qmul quant_mul)
{

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float inner_sum = 0.0f;
  float outer_sum = 0.0f;
  int currFactor = 0;

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
    for (int j = 0; j < blockDim.x; j++)
    {
      inner_sum = quant_add(inner_sum + quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx]));
    }
    currFactor++;
    currFactor %= BLOCK_FACTOR;
    if (currFactor == 0)
    {
      outer_sum = quant_add(outer_sum + inner_sum);
      inner_sum = 0.0f;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = outer_sum;
}

template <size_t BLOCK_FACTOR, size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void mm_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                        int *__restrict__ s,
                        int *__restrict__ mans,
                        int *__restrict__ exps,
                        int M, int K, int N,
                        Qadd quant_add, Qmul quant_mul)
{

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float inner_sum = 0.0f;
  float outer_sum = 0.0f;
  int currFactor = 0;

  int prec = 0;
  if (row < M && col < N)
    prec = s[row * N + col];

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
    for (int j = 0; j < blockDim.x; j++)
    {
      inner_sum = quant_add(inner_sum + quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], mans[prec], exps[prec]), mans[prec], exps[prec]);
    }
    currFactor++;
    currFactor %= BLOCK_FACTOR;
    if (currFactor == 0)
    {
      outer_sum = quant_add(outer_sum + inner_sum, mans[prec], exps[prec]);
      inner_sum = 0.0f;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = outer_sum;
}

template <size_t BLOCK_FACTOR, size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void bmm_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                         int M, int K, int N,
                         Qadd quant_add, Qmul quant_mul)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float inner_sum = 0.0f;
  float outer_sum = 0.0f;
  int currFactor = 0;

  // Determine the start index of the current batch in the 1D linearized arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++)
    {
      inner_sum = quant_add(inner_sum + quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx]));
    }
    currFactor++;
    currFactor %= BLOCK_FACTOR;
    if (currFactor == 0)
    {
      outer_sum = quant_add(outer_sum + inner_sum);
      inner_sum = 0.0f;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write the result back to global memory
  if (row < M && col < N)
  {
    c[batch_c + row * N + col] = outer_sum;
  }
}

template <size_t BLOCK_FACTOR, size_t SHMEM_SIZE, class Qfma>
__global__ void mm_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                            int M, int K, int N,
                            Qfma quant_fma)
{

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float inner_sum = 0.0f;
  float outer_sum = 0.0f;
  int currFactor = 0;

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
    for (int j = 0; j < blockDim.x; j++)
    {
      inner_sum = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], inner_sum));
    }
    currFactor++;
    currFactor %= BLOCK_FACTOR;
    if (currFactor == 0)
    {
      outer_sum = quant_fma(outer_sum + inner_sum);
      inner_sum = 0.0f;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = outer_sum;
}

template <size_t BLOCK_FACTOR, size_t SHMEM_SIZE, class Qfma>
__global__ void mm_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                            int *__restrict__ s,
                            int *__restrict__ mans,
                            int *__restrict__ exps,
                            int M, int K, int N,
                            Qfma quant_fma)
{

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float inner_sum = 0.0f;
  float outer_sum = 0.0f;
  int currFactor = 0;

  int prec = 0;
  if (row < M && col < N)
    prec = s[row * N + col];

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
    for (int j = 0; j < blockDim.x; j++)
    {
      inner_sum = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], inner_sum), mans[prec], exps[prec]);
    }
    currFactor++;
    currFactor %= BLOCK_FACTOR;
    if (currFactor == 0)
    {
      outer_sum = quant_fma(outer_sum + inner_sum, mans[prec], exps[prec]);
      inner_sum = 0.0f;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = outer_sum;
}

template <size_t BLOCK_FACTOR, size_t SHMEM_SIZE, class Qfma>
__global__ void bmm_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                             int M, int K, int N,
                             Qfma quant_fma)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float inner_sum = 0.0f;
  float outer_sum = 0.0f;
  int currFactor = 0;

  // Determine the start index of the current batch in the 1D linearized arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++)
    {
      inner_sum = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], inner_sum));
    }
    currFactor++;
    currFactor %= BLOCK_FACTOR;
    if (currFactor == 0)
    {
      outer_sum = quant_fma(outer_sum + inner_sum);
      inner_sum = 0.0f;
    }
    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write the result back to global memory
  if (row < M && col < N)
  {
    c[batch_c + row * N + col] = outer_sum;
  }
}

template <size_t SHMEM_SIZE, class RandType, class Qadd, class Qmul>
__global__ void mm_sr_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                           curandState_t *state,
                           int M, int K, int N,
                           Qadd quant_add, Qmul quant_mul)
{

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  RandType radd, rmul;

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
    for (int j = 0; j < blockDim.x; j++)
    {
      radd = gen_rand<RandType>(state, sidx);
      rmul = gen_rand<RandType>(state, sidx);
      tmp = quant_add(tmp + quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], rmul), radd);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = tmp;
}

template <size_t SHMEM_SIZE, class RandType, class Qadd, class Qmul>
__global__ void bmm_sr_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                            curandState_t *state,
                            int M, int K, int N,
                            Qadd quant_add, Qmul quant_mul)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  RandType radd, rmul;
  // Determine the start index of the current batch in the 1D linearized
  // arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++)
    {
      radd = gen_rand<RandType>(state, sidx);
      rmul = gen_rand<RandType>(state, sidx);
      tmp = quant_add(tmp + quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], rmul), radd);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[batch_c + row * N + col] = tmp;
}

template <size_t SHMEM_SIZE, class RandType, class Qfma>
__global__ void mm_sr_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                               curandState_t *state,
                               int M, int K, int N,
                               Qfma quant_fma)
{

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  RandType rfma;

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
    for (int j = 0; j < blockDim.x; j++)
    {
      rfma = gen_rand<RandType>(state, sidx);
      tmp = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp), rfma);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = tmp;
}

template <size_t SHMEM_SIZE, class RandType, class Qfma>
__global__ void bmm_sr_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                                curandState_t *state,
                                int M, int K, int N,
                                Qfma quant_fma)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  RandType rfma;

  // Determine the start index of the current batch in the 1D linearized
  // arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++)
    {
      rfma = gen_rand<RandType>(state, sidx);
      tmp = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp), rfma);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[batch_c + row * N + col] = tmp;
}

template <size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void mm_kahan_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                              int M, int K, int N,
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

template <size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void mm_kahan_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                              int *__restrict__ s,
                              int *__restrict__ mans,
                              int *__restrict__ exps,
                              int M, int K, int N,
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

  int prec = 0;
  if (row < M && col < N)
    prec = s[row * N + col];

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
      update = quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], mans[prec], exps[prec]);
      y = quant_add(update - comp_term, mans[prec], exps[prec]);
      t = quant_add(sum + y, mans[prec], exps[prec]);
      comp_term = quant_add(quant_add(t - sum, mans[prec], exps[prec]) - y, mans[prec], exps[prec]);
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

template <size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void bmm_kahan_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                               int M, int K, int N,
                               Qadd quant_add, Qmul quant_mul)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.0f;
  float comp_term = 0.0f;
  float update = 0.0;
  float y = 0.0f;
  float t = 0.0f;

  // Determine the start index of the current batch in the 1D linearized arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++)
    {
      update = quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx]);
      y = quant_add(update - comp_term);
      t = quant_add(sum + y);
      comp_term = quant_add(quant_add(t - sum) - y);
      sum = t;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write the result back to global memory
  if (row < M && col < N)
  {
    c[batch_c + row * N + col] = sum;
  }
}

template <size_t SHMEM_SIZE, class Qfma>
__global__ void mm_kahan_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                                  int M, int K, int N,
                                  Qfma quant_fma)
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
  float y = 0.0f;
  float t = 0.0f;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load the elements for this tile
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
      y = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], -comp_term));
      t = quant_fma(sum + y);
      comp_term = quant_fma(quant_fma(t - sum) - y);
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

template <size_t SHMEM_SIZE, class Qfma>
__global__ void mm_kahan_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                                  int *__restrict__ s,
                                  int *__restrict__ mans,
                                  int *__restrict__ exps,
                                  int M, int K, int N,
                                  Qfma quant_fma)
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
  float y = 0.0f;
  float t = 0.0f;

  int prec = 0;
  if (row < M && col < N)
    prec = s[row * N + col];

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load the elements for this tile
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
      y = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], -comp_term), mans[prec], exps[prec]);
      t = quant_fma(sum + y, mans[prec], exps[prec]);
      comp_term = quant_fma(quant_fma(t - sum, mans[prec], exps[prec]) - y, mans[prec], exps[prec]);
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

template <size_t SHMEM_SIZE, class Qfma>
__global__ void bmm_kahan_fma_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                                   int M, int K, int N,
                                   Qfma quant_fma)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int batch_idx = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.0f;
  float comp_term = 0.0f;
  float y = 0.0f;
  float t = 0.0f;

  // Determine the start index of the current batch in the 1D linearized arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
  {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++)
    {
      y = quant_fma(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], -comp_term));
      t = quant_fma(sum + y);
      comp_term = quant_fma(quant_fma(t - sum) - y);
      sum = t;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write the result back to global memory
  if (row < M && col < N)
  {
    c[batch_c + row * N + col] = sum;
  }
}