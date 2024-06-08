#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max)
    return max;
  else if (a < min)
    return min;
  else
    return a;
}

template <typename T>
__device__ __forceinline__ T clamp_mask_helper(T a, T min, T max,
                                               uint8_t *mask) {
  if (a > max) {
    *mask = 1;
    return max;
  } else if (a < min) {
    *mask = 1;
    return min;
  }
  *mask = 0;
  return a;
}

__device__ float cast_fxp_nearest(float origin_float, int sigma, float t_min,
                                  float t_max) {
  origin_float = nearest_round(origin_float, sigma);
  origin_float = clamp_helper(origin_float, t_min, t_max);
  return origin_float;
}

__device__ float cast_fxp_stochastic(float origin_float, float rand_prob,
                                     int sigma, float t_min, float t_max) {
  origin_float = round(origin_float, rand_prob, sigma);
  origin_float = clamp_helper(origin_float, t_min, t_max);
  return origin_float;
}

// quantize an array of real numbers into fixed point with word length [wl] and
// [fl] fractional bits 2**-[sigma] is the smallest unit of the fixed point
// representation. Stochastic Rounding with r.
__global__ void fixed_point_quantize_kernel_stochastic(
    float *__restrict__ a, float *__restrict__ r, float *o, int size, int sigma,
    bool use_clamp, float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

// quantize an array of real numbers into fixed point with word length [wl] and
// [fl] fractional bits 2**-[sigma] is the smallest unit of the fixed point
// representation. Nearest Neighbor Rounding.
__global__ void fixed_point_quantize_kernel_nearest(float *__restrict__ a,
                                                    float *o, int size,
                                                    int sigma, bool use_clamp,
                                                    float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

__global__ void fixed_point_quantize_kernel_mask_stochastic(
    float *__restrict__ a, float *__restrict__ r, float *o, uint8_t *m,
    int size, int sigma, float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m + index);
  }
}

__global__ void fixed_point_quantize_kernel_mask_nearest(float *__restrict__ a,
                                                         float *o, uint8_t *m,
                                                         int size, int sigma,
                                                         float t_min,
                                                         float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m + index);
  }
}

template <size_t SHMEM_SIZE>
__global__ void
mm_fxp_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                    float *__restrict__ c, int M, int K, int N, int sigma_add,
                    int t_min_add, int t_max_add, int sigma_mul, int t_min_mul,
                    int t_max_mul) {
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
      tmp = cast_fxp_nearest(
          tmp + cast_fxp_nearest(s_a[ty * blockDim.x + j] *
                                     s_b[j * blockDim.x + tx],
                                 sigma_mul, t_min_mul, t_max_mul),
          sigma_add, t_min_add, t_max_add);
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
bmm_fxp_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                     float *__restrict__ c, int M, int K, int N, int sigma_add,
                     int t_min_add, int t_max_add, int sigma_mul, int t_min_mul,
                     int t_max_mul) {
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
      tmp = cast_fxp_nearest(
          tmp + cast_fxp_nearest(s_a[ty * blockDim.x + j] *
                                     s_b[j * blockDim.x + tx],
                                 sigma_mul, t_min_mul, t_max_mul),
          sigma_add, t_min_add, t_max_add);
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
mm_fxp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                        float *__restrict__ c, int M, int K, int N,
                        int sigma_fma, int t_min_fma, int t_max_fma) {
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
      tmp = cast_fxp_nearest(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
          sigma_fma, t_min_fma, t_max_fma);
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
bmm_fxp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b,
                         float *__restrict__ c, int M, int K, int N,
                         int sigma_fma, int t_min_fma, int t_max_fma) {
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
      tmp = cast_fxp_nearest(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
          sigma_fma, t_min_fma, t_max_fma);
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
__global__ void mm_fxp_stochastic_impl(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    curandState_t *state, // float *__restrict__ r,
    int M, int K, int N, int sigma_add, int t_min_add, int t_max_add,
    int sigma_mul, int t_min_mul, int t_max_mul) {
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
  float radd, rmul;

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
      radd = 1.0f - curand_uniform(&state[sidx]);
      rmul = 1.0f - curand_uniform(&state[sidx]);
      // radd = r[bidx + 2 * (i + j)];
      // rmul = r[bidx + 2 * (i + j) + 1];
      tmp = cast_fxp_stochastic(
          tmp + cast_fxp_stochastic(s_a[ty * blockDim.x + j] *
                                        s_b[j * blockDim.x + tx],
                                    rmul, sigma_mul, t_min_mul, t_max_mul),
          radd, sigma_add, t_min_add, t_max_add);
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
__global__ void bmm_fxp_stochastic_impl(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    curandState_t *state, // float *__restrict__ r,
    int M, int K, int N, int sigma_add, int t_min_add, int t_max_add,
    int sigma_mul, int t_min_mul, int t_max_mul) {
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
  float radd, rmul;

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
      radd = 1.0f - curand_uniform(&state[sidx]);
      rmul = 1.0f - curand_uniform(&state[sidx]);
      // radd = r[bidx + 2 * (i + j)];
      // rmul = r[bidx + 2 * (i + j) + 1];
      tmp = cast_fxp_stochastic(
          tmp + cast_fxp_stochastic(s_a[ty * blockDim.x + j] *
                                        s_b[j * blockDim.x + tx],
                                    rmul, sigma_mul, t_min_mul, t_max_mul),
          radd, sigma_add, t_min_add, t_max_add);
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
__global__ void mm_fxp_fma_stochastic_impl(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    curandState_t *state, // float *__restrict__ r,
    int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x);
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0;
  float rfma;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      // rfma = r[bidx + i + j];
      rfma = 1.0f - curand_uniform(&state[sidx]);
      tmp = cast_fxp_stochastic(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp), rfma,
          sigma_fma, t_min_fma, t_max_fma);
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
__global__ void bmm_fxp_fma_stochastic_impl(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    curandState_t *state, // float *__restrict__ r,
    int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
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

  float tmp = 0.0;
  float rfma;

  // Determine the start index of the current batch in the 1D linearized arrays
  int batch_a = batch_idx * M * K;
  int batch_b = batch_idx * K * N;
  int batch_c = batch_idx * M * N;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[batch_a + row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[batch_b + i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      // rfma = r[bidx + i + j];
      rfma = 1.0f - curand_uniform(&state[sidx]);
      tmp = cast_fxp_stochastic(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp), rfma,
          sigma_fma, t_min_fma, t_max_fma);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[batch_c + row * N + col] = tmp;
}

void mm_fxp_nearest(float *a, float *b, float *c, int M, int K, int N,
                    int sigma_add, int t_min_add, int t_max_add, int sigma_mul,
                    int t_min_mul, int t_max_mul) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_fxp_nearest_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, sigma_add, t_min_add,
                                  t_max_add, sigma_mul, t_min_mul, t_max_mul);
}

void bmm_fxp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                     int sigma_add, int t_min_add, int t_max_add, int sigma_mul,
                     int t_min_mul, int t_max_mul) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  bmm_fxp_nearest_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N, sigma_add, t_min_add,
                                  t_max_add, sigma_mul, t_min_mul, t_max_mul);
}

void mm_fxp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                        int sigma_fma, int t_min_fma, int t_max_fma) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  mm_fxp_fma_nearest_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, sigma_fma, t_min_fma, t_max_fma);
}

void bmm_fxp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                         int N, int sigma_fma, int t_min_fma, int t_max_fma) {

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  bmm_fxp_fma_nearest_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, sigma_fma, t_min_fma, t_max_fma);
}

void mm_fxp_stochastic(float *a, float *b, float *c, int M, int K, int N,
                       int sigma_add, int t_min_add, int t_max_add,
                       int sigma_mul, int t_min_mul, int t_max_mul) {
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
  // TODO: change this to a fixed seed?!
  seed_init<<<block_dim, 1>>>(time(0), state);
  mm_fxp_stochastic_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, state, M, K, N, sigma_add, t_min_add,
                                  t_max_add, sigma_mul, t_min_mul, t_max_mul);
  cudaFree(state);
}

void bmm_fxp_stochastic(float *a, float *b, float *c, int B, int M, int K,
                        int N, int sigma_add, int t_min_add, int t_max_add,
                        int sigma_mul, int t_min_mul, int t_max_mul) {
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  // TODO: change this to a fixed seed?!
  seed_init<<<block_dim, 1>>>(time(0), state);
  bmm_fxp_stochastic_impl<SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, state, M, K, N, sigma_add, t_min_add,
                                  t_max_add, sigma_mul, t_min_mul, t_max_mul);
  cudaFree(state);
}

void mm_fxp_fma_stochastic(float *a, float *b, float *c, int M, int K, int N,
                           int sigma_fma, int t_min_fma, int t_max_fma) {
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
  // TODO: change this to a fixed seed?!
  seed_init<<<block_dim, 1>>>(time(0), state);
  mm_fxp_fma_stochastic_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, state, M, K, N, sigma_fma, t_min_fma, t_max_fma);
  cudaFree(state);
}

void bmm_fxp_fma_stochastic(float *a, float *b, float *c, int B, int M, int K,
                            int N, int sigma_fma, int t_min_fma,
                            int t_max_fma) {
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  // TODO: change this to a fixed seed?!
  seed_init<<<block_dim, 1>>>(time(0), state);
  bmm_fxp_fma_stochastic_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, state, M, K, N, sigma_fma, t_min_fma, t_max_fma);
  cudaFree(state);
}