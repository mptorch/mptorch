#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

template <bool subnormals>
__device__ float cast_p3109_signed_nearest(float origin_float, int P) {
  // TODO
  return 0.0f;
}

template <>
__device__ float cast_p3109_signed_nearest<false>(float origin_float, int P) {
  // TODO:
  return 0.0f;
}

template <bool subnormals>
__device__ float cast_p3109_signed_stochastic(float origin_float, int P, int prng_bits) {
  // TODO:
  return 0.0f;
}

template <>
__device__ float cast_p3109_signed_stochastic<false>(float origin_float, int P, int prng_bits) {
  // TODO:
  return 0.0f;
}

template <bool subnormals>
__device__ float cast_p3109_unsigned_nearest(float origin_float, int P) {
  // TODO:
  return 0.0f;
}

template <>
__device__ float cast_p3109_unsigned_nearest<false>(float origin_float, int P) {
  // TODO:
  return 0.0f;
}

template <bool subnormals>
__device__ float cast_p3109_unsigned_stochastic(float origin_float, int P, int prng_bits) {
  // TODO:
  return 0.0f;
}

template <>
__device__ float cast_p3109_unsigned_stochastic<false>(float origin_float, int P, int prng_bits) {
  // TODO:
  return 0.0f;
}

__global__ void p3109_signed_kernel_nearest(float *__restrict__ a, float *o, int size,
                                      int P, bool subnormals) {
  // TODO
}

__global__ void p3109_unsigned_kernel_nearest(float *__restrict__ a, float *o, int size,
                                      int P, bool subnormals) {
  // TODO
}

__global__ void p3109_signed_kernel_stochastic(float *__restrict__ a, float *o, int size,
                                      int P, int prng_bits, bool subnormals) {
  // TODO
}

__global__ void p3109_unsigned_kernel_stochastic(float *__restrict__ a, float *o, int size,
                                      int P, int prng_bits, bool subnormals) {
  // TODO
}