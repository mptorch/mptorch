#include "quant_kernel.h"
#include "sim_helper.cu"
#include "mm_kernel.h"
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
  mm_impl<1u, SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N,
      [sigma_add, t_min_add, t_max_add] __device__ (float x) { return cast_fxp_nearest(x, sigma_add, t_min_add, t_max_add); },
      [sigma_mul, t_min_mul, t_max_mul] __device__ (float x) { return cast_fxp_nearest(x, sigma_mul, t_min_mul, t_max_mul); });
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
  bmm_impl<1u, SHMEM_SIZE>
      <<<block_dim, thread_dim>>>(a, b, c, M, K, N,
      [sigma_add, t_min_add, t_max_add] __device__ (float x) { return cast_fxp_nearest(x, sigma_add, t_min_add, t_max_add); },
      [sigma_mul, t_min_mul, t_max_mul] __device__ (float x) { return cast_fxp_nearest(x, sigma_mul, t_min_mul, t_max_mul); });
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
  mm_fma_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N, 
      [sigma_fma, t_min_fma, t_max_fma] __device__ (float x) { return cast_fxp_nearest(x, sigma_fma, t_min_fma, t_max_fma); });
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
  bmm_fma_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
      a, b, c, M, K, N,
      [sigma_fma, t_min_fma, t_max_fma] __device__ (float x) { return cast_fxp_nearest(x, sigma_fma, t_min_fma, t_max_fma); });
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
  seed_init<<<block_dim, 1>>>(state);
  mm_sr_impl<SHMEM_SIZE, float>
      <<<block_dim, thread_dim>>>(a, b, c, state, M, K, N,
      [sigma_add, t_min_add, t_max_add] __device__ (float x, float rand_prob) { return cast_fxp_stochastic(x, rand_prob, sigma_add, t_min_add, t_max_add); },
      [sigma_mul, t_min_mul, t_max_mul] __device__ (float x, float rand_prob) { return cast_fxp_stochastic(x, rand_prob, sigma_mul, t_min_mul, t_max_mul); }
  );
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
  seed_init<<<block_dim, 1>>>(state);
  bmm_sr_impl<SHMEM_SIZE, float>
      <<<block_dim, thread_dim>>>(a, b, c, state, M, K, N,
      [sigma_add, t_min_add, t_max_add] __device__ (float x, float rand_prob) { return cast_fxp_stochastic(x, rand_prob, sigma_add, t_min_add, t_max_add); },
      [sigma_mul, t_min_mul, t_max_mul] __device__ (float x, float rand_prob) { return cast_fxp_stochastic(x, rand_prob, sigma_mul, t_min_mul, t_max_mul); }
  );
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
  seed_init<<<block_dim, 1>>>(state);
  mm_sr_fma_impl<SHMEM_SIZE, float><<<block_dim, thread_dim>>>(
      a, b, c, state, M, K, N,
      [sigma_fma, t_min_fma, t_max_fma] __device__ (float x, float rand_prob) { return cast_fxp_stochastic(x, rand_prob, sigma_fma, t_min_fma, t_max_fma); }
  );
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
  seed_init<<<block_dim, 1>>>(state);
  bmm_sr_fma_impl<SHMEM_SIZE, float><<<block_dim, thread_dim>>>(
      a, b, c, state, M, K, N,
      [sigma_fma, t_min_fma, t_max_fma] __device__ (float x, float rand_prob) { return cast_fxp_stochastic(x, rand_prob, sigma_fma, t_min_fma, t_max_fma); }
  );
  cudaFree(state);
}