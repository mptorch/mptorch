#include "quant_kernel.h"
#include "sim_helper.cu"


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
__device__ __forceinline__ T clamp_mask_helper(T a, T min, T max, uint8_t* mask) {
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

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Stochastic Rounding with r.
__global__ void fixed_point_quantize_kernel_stochastic(float* __restrict__ a,
                                                       float* __restrict__ r,
                                                       float* o, int size,
                                                       int sigma, bool use_clamp,
                                                       float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Nearest Neighbor Rounding.
__global__ void fixed_point_quantize_kernel_nearest(float* __restrict__ a,
                                                    float* o, int size,
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

__global__ void fixed_point_quantize_kernel_mask_stochastic(float* __restrict__ a,
                                                            float* __restrict__ r,
                                                            float* o, uint8_t* m,
                                                            int size, int sigma,
                                                            float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}

__global__ void fixed_point_quantize_kernel_mask_nearest(float* __restrict__ a,
                                                         float* o, uint8_t* m,
                                                         int size, int sigma,
                                                         float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}

__global__ void gemm_fxp_nearest(float *__restrict__ a, float *__restrict__ b,
                          float *__restrict__ c, int M, int K, int N,
                          int sigma_add, int t_min_add, int t_max_add,
                          int sigma_mul, int t_min_mul, int t_max_mul)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
  // load in elements for this tile
    s_a[ty * blockDim.x + tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] = (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_fxp_nearest(tmp + cast_fxp_nearest(s_a[ty * blockDim.x + j] * 
                  s_b[j * blockDim.x + tx], sigma_mul, t_min_mul, t_max_mul),
                  sigma_add, t_min_add, t_max_add);
    }

    // wait for all threads to finish using current tiles 
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N) c[row * N + col] = tmp;
}

__global__ void gemm_fxp_fma_nearest(float *__restrict__ a, float *__restrict__ b,
                          float *__restrict__ c, int M, int K, int N,
                          int sigma_fma, int t_min_fma, int t_max_fma)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
  // load in elements for this tile
    s_a[ty * blockDim.x + tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] = (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_fxp_nearest(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
                  sigma_fma, t_min_fma, t_max_fma);
    }

    // wait for all threads to finish using current tiles 
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N) c[row * N + col] = tmp;
}

__global__ void gemm_fxp_stochastic(float *__restrict__ a, float *__restrict__ b,
                          float *__restrict__ c, curandState_t *state, // float *__restrict__ r,
                          int M, int K, int N,
                          int sigma_add, int t_min_add, int t_max_add,
                          int sigma_mul, int t_min_mul, int t_max_mul)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  //int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x) * 2;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0;
  float radd, rmul;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
  // load in elements for this tile
    s_a[ty * blockDim.x + tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] = (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      radd = 1.0f - curand_uniform(&state[sidx]);
      rmul = 1.0f - curand_uniform(&state[sidx]);
      //radd = r[bidx + 2 * (i + j)];
      //rmul = r[bidx + 2 * (i + j) + 1];
      tmp = cast_fxp_stochastic(tmp + cast_fxp_stochastic(s_a[ty * blockDim.x + j] * 
                  s_b[j * blockDim.x + tx], rmul, sigma_mul, t_min_mul, t_max_mul),
                  radd, sigma_add, t_min_add, t_max_add);
    }

    // wait for all threads to finish using current tiles 
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N) c[row * N + col] = tmp;
}

__global__ void gemm_fxp_fma_stochastic(float *__restrict__ a, float *__restrict__ b,
                          float *__restrict__ c, curandState_t *state, // float *__restrict__ r, 
                          int M, int K, int N,
                          int sigma_fma, int t_min_fma, int t_max_fma)
{
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

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
    s_a[ty * blockDim.x + tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] = (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      // rfma = r[bidx + i + j];
      rfma = 1.0f - curand_uniform(&state[sidx]);
      tmp = cast_fxp_stochastic(fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
                  rfma, sigma_fma, t_min_fma, t_max_fma);
    }

    // wait for all threads to finish using current tiles 
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N) c[row * N + col] = tmp;
}