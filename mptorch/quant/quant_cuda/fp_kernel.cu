#include "bit_helper.cu"
#include "modes.h"
#include "quant_kernel.h"
#include "mm_kernel.h"
#include "sim_helper.cu"
#include "layernorm_kernel.h"
#include "softmax_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float cast_fp_nearest_even(float origin_float,
                                      int man_bits, int exp_bits,
                                      int bias,
                                      bool saturate,
                                      SubnormalsMode subnormals)
{
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -(bias - 1);
  bool subnormal = (target_exp < min_exp);
  bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

  if (noquantize)
  {
    quantized = origin_float;
  }
  else
  {
    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
      int exp_diff = man_bits - (min_exp - target_exp);
      int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
      quantize_bits = not_uflow * round_bitwise_nearest_even(target, exp_diff);
      quantize_bits =
          clip_subnormal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits);
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
      quantize_bits = round_bitwise_nearest_even(target, man_bits);
      quantize_bits = clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                                 saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
  }

  return quantized;
}

__device__ float cast_fp_nearest_away(float origin_float,
                                      int man_bits, int exp_bits, int bias,
                                      bool saturate,
                                      SubnormalsMode subnormals)
{
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -(bias - 1);
  bool subnormal = (target_exp < min_exp);
  bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

  if (noquantize)
  {
    quantized = origin_float;
  }
  else
  {
    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
      int exp_diff = man_bits - (min_exp - target_exp);
      int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
      quantize_bits = not_uflow * round_bitwise_nearest_away(target, exp_diff);
      quantize_bits =
          clip_subnormal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits);
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
      quantize_bits = round_bitwise_nearest_away(target, man_bits);
      quantize_bits =
          clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                     saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
  }

  return quantized;
}

__device__ float cast_absolute_up(float origin_float,
                                  int man_bits, int exp_bits, int bias,
                                  bool saturate,
                                  SubnormalsMode subnormals)
{
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -bias + 1;
  bool subnormal = (target_exp < min_exp);
  bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

  if (noquantize)
  {
    quantized = origin_float;
  }
  else
  {
    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
      int exp_diff = man_bits - (min_exp - target_exp);
      int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
      quantize_bits = not_uflow * round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
      quantize_bits =
          clip_subnormal_range_exponent_up(exp_bits, man_bits, bias, target, quantize_bits);
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
      quantize_bits = round_bitwise_up(target, man_bits);
      quantize_bits =
          clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                     saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
  }

  return quantized;
}

__device__ float cast_absolute_down(float origin_float,
                                    int man_bits, int exp_bits, int bias,
                                    bool saturate,
                                    SubnormalsMode subnormals)
{
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -bias + 1;
  bool subnormal = (target_exp < min_exp);
  bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

  if (noquantize)
  {
    quantized = origin_float;
  }
  else
  {
    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
      int exp_diff = man_bits - (min_exp - target_exp);
      int not_uflow = exp_diff > -1;
      quantize_bits = not_uflow * round_bitwise_down(target, exp_diff);
      quantize_bits =
          clip_subnormal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits);
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
      quantize_bits = round_bitwise_down(target, man_bits);
      quantize_bits =
          clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                     saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
  }

  return quantized;
}

__device__ float cast_fp_up(float origin_float,
                            int man_bits, int exp_bits, int bias,
                            bool saturate,
                            SubnormalsMode subnormals)
{
  if (origin_float >= 0.0f)
    return cast_absolute_up(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
  else
    return -cast_absolute_down(-origin_float, man_bits, exp_bits, bias, saturate, subnormals);
}

__device__ float cast_fp_down(float origin_float,
                              int man_bits, int exp_bits, int bias,
                              bool saturate,
                              SubnormalsMode subnormals)
{
  if (origin_float >= 0.0f)
    return cast_absolute_down(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
  else
    return -cast_absolute_up(-origin_float, man_bits, exp_bits, bias, saturate, subnormals);
}

__device__ float cast_fp_zero(float origin_float,
                              int man_bits, int exp_bits, int bias,
                              bool saturate,
                              SubnormalsMode subnormals)
{
  if (origin_float >= 0.0f)
    return cast_fp_down(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
  else
    return cast_fp_up(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
}

__device__ float cast_fp_stochastic(float origin_float, uint32_t rand_prob,
                                    int man_bits, int exp_bits, int bias,
                                    bool saturate,
                                    SubnormalsMode subnormals)
{
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -(bias - 1);
  bool subnormal = (target_exp < min_exp);

  if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
  {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    shift_float = BITS_TO_FLOAT(&shift_bits);
    val = origin_float + shift_float;
    target = FLOAT_TO_BITS(&val);
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
  }
  else
  {
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantize_bits = clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                               saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
    quantized = BITS_TO_FLOAT(&quantize_bits);
  }

  return quantized;
}

__device__ float cast_fp_stochastic(float origin_float, uint32_t rand_prob,
                                    int rand_bits, int man_bits, int exp_bits, int bias,
                                    bool saturate,
                                    SubnormalsMode subnormals)
{
  uint32_t target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -(bias - 1);
  bool subnormal = (target_exp < min_exp);

  rand_prob = rand_prob << 9 >> 9;
  rand_prob = rand_prob & ~((1 << (23 - man_bits - rand_bits)) - 1);

  if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
  {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    shift_float = BITS_TO_FLOAT(&shift_bits);
    val = origin_float + shift_float;
    target = FLOAT_TO_BITS(&val);
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
  }
  else
  {
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantize_bits = clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                               saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
    quantized = BITS_TO_FLOAT(&quantize_bits);
  }

  return quantized;
}

__global__ void seed_init(curandState_t *state)
{
  curand_init(clock64(), blockIdx.x * blockIdx.y, 0,
              &state[blockIdx.x * blockIdx.y]);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits,
                                        bool saturate,
                                        SubnormalsMode subnormals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias = (1 << (exp_bits - 1)) - 1;
  if (index < size)
    o[index] = cast_fp_stochastic(a[index], (uint32_t)r[index],
                                  man_bits, exp_bits, bias,
                                  saturate, subnormals);
}

__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits, int prng_bits,
                                        bool saturate,
                                        SubnormalsMode subnormals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias = (1 << (exp_bits - 1)) - 1;
  if (index < size)
    o[index] = cast_fp_stochastic(a[index], (uint32_t)r[index],
                                  prng_bits, man_bits, exp_bits, bias,
                                  saturate, subnormals);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa using round to nearest ties to even
__global__ void float_kernel_nearest_even(float *__restrict__ a, float *o, int size,
                                          int man_bits, int exp_bits,
                                          bool saturate,
                                          SubnormalsMode subnormals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias = (1 << (exp_bits - 1)) - 1;
  if (index < size)
    o[index] = cast_fp_nearest_even(a[index], man_bits, exp_bits, bias, saturate, subnormals);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa using round to nearest ties to away
__global__ void float_kernel_nearest_away(float *__restrict__ a, float *o, int size,
                                          int man_bits, int exp_bits,
                                          bool saturate,
                                          SubnormalsMode subnormals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias = (1 << (exp_bits - 1)) - 1;
  if (index < size)
    o[index] = cast_fp_nearest_away(a[index], man_bits, exp_bits, bias, saturate, subnormals);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa using round up
__global__ void float_kernel_up(float *__restrict__ a, float *o, int size,
                                int man_bits, int exp_bits,
                                bool saturate,
                                SubnormalsMode subnormals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias = (1 << (exp_bits - 1)) - 1;
  if (index < size)
    o[index] = cast_fp_up(a[index], man_bits, exp_bits, bias, saturate, subnormals);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa using round up
__global__ void float_kernel_down(float *__restrict__ a, float *o, int size,
                                  int man_bits, int exp_bits,
                                  bool saturate,
                                  SubnormalsMode subnormals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias = (1 << (exp_bits - 1)) - 1;
  if (index < size)
    o[index] = cast_fp_down(a[index], man_bits, exp_bits, bias, saturate, subnormals);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa using round towards zero
__global__ void float_kernel_zero(float *__restrict__ a, float *o, int size,
                                  int man_bits, int exp_bits,
                                  bool saturate,
                                  SubnormalsMode subnormals)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias = (1 << (exp_bits - 1)) - 1;
  if (index < size)
    o[index] = cast_fp_zero(a[index], man_bits, exp_bits, bias, saturate, subnormals);
}

void fp_kernel(float *__restrict__ a, float *o, int size,
               int man_bits, int exp_bits, int bias,
               bool saturate,
               RoundMode round_mode,
               SubnormalsMode subnormals)
{
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  switch (round_mode)
  {
  case RoundMode::RNE:
    quant_kernel<<<blockNums, blockSize>>>(
        a, o, size, [man_bits, exp_bits, bias, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_bits, exp_bits, bias, saturate, subnormals); });
    break;

  case RoundMode::RNA:
    quant_kernel<<<blockNums, blockSize>>>(
        a, o, size, [man_bits, exp_bits, bias, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_away(x, man_bits, exp_bits, bias, saturate, subnormals); });
    break;

  case RoundMode::RU:
    quant_kernel<<<blockNums, blockSize>>>(
        a, o, size, [man_bits, exp_bits, bias, saturate, subnormals] __device__(float x)
        { return cast_fp_up(x, man_bits, exp_bits, bias, saturate, subnormals); });
    break;

  case RoundMode::RD:
    quant_kernel<<<blockNums, blockSize>>>(
        a, o, size, [man_bits, exp_bits, bias, saturate, subnormals] __device__(float x)
        { return cast_fp_down(x, man_bits, exp_bits, bias, saturate, subnormals); });
    break;

  default: // RZ
    quant_kernel<<<blockNums, blockSize>>>(
        a, o, size, [man_bits, exp_bits, bias, saturate, subnormals] __device__(float x)
        { return cast_fp_zero(x, man_bits, exp_bits, bias, saturate, subnormals); });
    break;
  }
}

void fp_kernel(float *__restrict__ a, int *__restrict__ r, float *o, int size,
               int man_bits, int exp_bits, int bias, int prng_bits,
               bool saturate,
               RoundMode round_mode,
               SubnormalsMode subnormals)
{
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  quant_kernel<<<blockNums, blockSize>>>(
      a, r, o, size, [man_bits, exp_bits, prng_bits, bias, saturate, subnormals] __device__(float x, uint32_t rv)
      { return cast_fp_stochastic(x, rv, prng_bits, man_bits, exp_bits, bias, saturate, subnormals); });
}

void mm_fp_nearest(float *a, float *b, float *c,
                   int M, int K, int N,
                   int man_add, int exp_add,
                   int man_mul, int exp_mul,
                   bool saturate,
                   SubnormalsMode subnormals,
                   bool compensated)
{

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  int bias_add = (1 << (exp_add - 1)) - 1;
  int bias_mul = (1 << (exp_mul - 1)) - 1;
  if (compensated)
  {
    mm_kahan_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_add, exp_add, bias_add, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_add, exp_add, bias_add, saturate, subnormals); },
        [man_mul, exp_mul, bias_mul, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_mul, exp_mul, bias_mul, saturate, subnormals); });
  }
  else
  {
    mm_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_add, exp_add, bias_add, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_add, exp_add, bias_add, saturate, subnormals); },
        [man_mul, exp_mul, bias_mul, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_mul, exp_mul, bias_mul, saturate, subnormals); });
  }
}

void bmm_fp_nearest(float *a, float *b, float *c,
                    int B, int M, int K, int N,
                    int man_add, int exp_add,
                    int man_mul, int exp_mul,
                    bool saturate,
                    SubnormalsMode subnormals,
                    bool compensated)
{

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  int bias_add = (1 << (exp_add - 1)) - 1;
  int bias_mul = (1 << (exp_mul - 1)) - 1;
  if (compensated)
  {
    bmm_kahan_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_add, exp_add, bias_add, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_add, exp_add, bias_add, saturate, subnormals); },
        [man_mul, exp_mul, bias_mul, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_mul, exp_mul, bias_mul, saturate, subnormals); });
  }
  else
  {
    bmm_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_add, exp_add, bias_add, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_add, exp_add, bias_add, saturate, subnormals); },
        [man_mul, exp_mul, bias_mul, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_mul, exp_mul, bias_mul, saturate, subnormals); });
  }
}

void mm_fp_fma_nearest(float *a, float *b, float *c,
                       int M, int K, int N,
                       int man_fma, int exp_fma,
                       bool saturate,
                       SubnormalsMode subnormals,
                       bool compensated)
{
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  int bias_fma = (1 << (exp_fma - 1)) - 1;
  if (compensated)
  {
    mm_kahan_fma_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_fma, exp_fma, bias_fma, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_fma, exp_fma, bias_fma, saturate, subnormals); });
  }
  else
  {
    mm_fma_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_fma, exp_fma, bias_fma, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_fma, exp_fma, bias_fma, saturate, subnormals); });
  }
}

void bmm_fp_fma_nearest(float *a, float *b, float *c,
                        int B, int M, int K, int N,
                        int man_fma, int exp_fma,
                        bool saturate,
                        SubnormalsMode subnormals,
                        bool compensated)
{

  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  int bias_fma = (1 << (exp_fma - 1)) - 1;
  if (compensated)
  {
    bmm_kahan_fma_impl<SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_fma, exp_fma, bias_fma, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_fma, exp_fma, bias_fma, saturate, subnormals); });
  }
  else
  {
    bmm_fma_impl<1u, SHMEM_SIZE><<<block_dim, thread_dim>>>(
        a, b, c, M, K, N,
        [man_fma, exp_fma, bias_fma, saturate, subnormals] __device__(float x)
        { return cast_fp_nearest_even(x, man_fma, exp_fma, bias_fma, saturate, subnormals); });
  }
}

void mm_fp_stochastic(float *a, float *b, float *c,
                      int M, int K, int N,
                      int man_add, int exp_add, int rbits_add,
                      int man_mul, int exp_mul, int rbits_mul,
                      bool saturate,
                      SubnormalsMode subnormals)
{
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  int bias_add = (1 << (exp_add - 1)) - 1;
  int bias_mul = (1 << (exp_mul - 1)) - 1;
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  mm_sr_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state,
      M, K, N,
      [man_add, exp_add, bias_add, rbits_add, saturate, subnormals] __device__(float x, uint32_t rnd)
      { return cast_fp_stochastic(x, rnd, rbits_add, man_add, exp_add, bias_add, saturate, subnormals); },
      [man_mul, exp_mul, bias_mul, rbits_mul, saturate, subnormals] __device__(float x, uint32_t rnd)
      { return cast_fp_stochastic(x, rnd, rbits_mul, man_mul, exp_mul, bias_mul, saturate, subnormals); });
  cudaFree(state);
}

void bmm_fp_stochastic(float *a, float *b, float *c,
                       int B, int M, int K, int N,
                       int man_add, int exp_add, int rbits_add,
                       int man_mul, int exp_mul, int rbits_mul,
                       bool saturate,
                       SubnormalsMode subnormals)
{
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  int bias_add = (1 << (exp_add - 1)) - 1;
  int bias_mul = (1 << (exp_mul - 1)) - 1;
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  bmm_sr_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state,
      M, K, N,
      [man_add, exp_add, bias_add, rbits_add, saturate, subnormals] __device__(float x, uint32_t rnd)
      { return cast_fp_stochastic(x, rnd, rbits_add, man_add, exp_add, bias_add, saturate, subnormals); },
      [man_mul, exp_mul, bias_mul, rbits_mul, saturate, subnormals] __device__(float x, uint32_t rnd)
      { return cast_fp_stochastic(x, rnd, rbits_mul, man_mul, exp_mul, bias_mul, saturate, subnormals); });
  cudaFree(state);
}

void mm_fp_fma_stochastic(float *a, float *b, float *c,
                          int M, int K, int N,
                          int man_fma, int exp_fma, int rbits_fma,
                          bool saturate,
                          SubnormalsMode subnormals)
{
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 const block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
  int bias_fma = (1 << (exp_fma - 1)) - 1;
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  mm_sr_fma_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state,
      M, K, N,
      [man_fma, exp_fma, bias_fma, rbits_fma, saturate, subnormals] __device__(float x, uint32_t rnd)
      { return cast_fp_stochastic(x, rnd, rbits_fma, man_fma, exp_fma, bias_fma, saturate, subnormals); });
  cudaFree(state);
}

void bmm_fp_fma_stochastic(float *a, float *b, float *c,
                           int B, int M, int K, int N,
                           int man_fma, int exp_fma, int rbits_fma,
                           bool saturate,
                           SubnormalsMode subnormals)
{
  constexpr size_t THREADS_X{8U};
  constexpr size_t THREADS_Y{8U};
  constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
  dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
  dim3 block_dim{
      (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
      (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y,
      static_cast<uint32_t>(B)};
  int bias_fma = (1 << (exp_fma - 1)) - 1;
  curandState_t *state;
  cudaMalloc((void **)&state,
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(state);
  bmm_sr_fma_impl<SHMEM_SIZE, uint32_t><<<block_dim, thread_dim>>>(
      a, b, c,
      state,
      M, K, N,
      [man_fma, exp_fma, bias_fma, rbits_fma, saturate, subnormals] __device__(float x, uint32_t rnd)
      { return cast_fp_stochastic(x, rnd, rbits_fma, man_fma, exp_fma, bias_fma, saturate, subnormals); });
  cudaFree(state);
}

void softmax_forward_fp_nearest(float *a, float *o,
                                const DimSizes &sizes,
                                int man_exp, int exp_exp,
                                int man_off, int exp_off,
                                int man_acc, int exp_acc,
                                bool saturate,
                                SubnormalsMode subnormals)
{
  int bias_exp = (1 << (exp_exp - 1)) - 1;
  int bias_off = (1 << (exp_off - 1)) - 1;
  int bias_acc = (1 << (exp_acc - 1)) - 1;
  softmax_forward(a, o, sizes, [man_exp, exp_exp, bias_exp, saturate, subnormals] __device__(float x)
                  { return cast_fp_nearest_even(x, man_exp, exp_exp, bias_exp, saturate, subnormals); }, [man_off, exp_off, bias_off, saturate, subnormals] __device__(float x)
                  { return cast_fp_nearest_even(x, man_off, exp_off, bias_off, saturate, subnormals); }, [man_acc, exp_acc, bias_acc, saturate, subnormals] __device__(float x)
                  { return cast_fp_nearest_even(x, man_acc, exp_acc, bias_acc, saturate, subnormals); });
}

void softmax_lse_forward_fp_nearest(float *a, float *o,
                                    const DimSizes &sizes,
                                    int man_off, int exp_off,
                                    int man_lse, int exp_lse,
                                    bool saturate,
                                    SubnormalsMode subnormals)
{
  int bias_off = (1 << (exp_off - 1)) - 1;
  int bias_lse = (1 << (exp_lse - 1)) - 1;
  softmax_lse_forward(a, o, sizes, [man_off, exp_off, bias_off, saturate, subnormals] __device__(float x)
                      { return cast_fp_nearest_even(x, man_off, exp_off, bias_off, saturate, subnormals); }, [man_lse, exp_lse, bias_lse, saturate, subnormals] __device__(float x)
                      { return cast_fp_nearest_even(x, man_lse, exp_lse, bias_lse, saturate, subnormals); });
}

void softmax_backward_fp_nearest(float *a, float *g, float *o,
                                 const DimSizes &sizes,
                                 int man_add, int exp_add,
                                 int man_mul, int exp_mul,
                                 bool saturate,
                                 SubnormalsMode subnormals)
{
  int bias_add = (1 << (exp_add - 1)) - 1;
  int bias_mul = (1 << (exp_mul - 1)) - 1;
  softmax_backward(
      a, g, o, sizes,
      [man_add, exp_add, bias_add, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_add, exp_add, bias_add, saturate, subnormals); },
      [man_mul, exp_mul, bias_mul, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_mul, exp_mul, bias_mul, saturate, subnormals); });
}

void layernorm_forward_fp_nearest(float *input, float *weight, float *bias,
                                  float *output, float *mean, float *rstd,
                                  float eps, const DimSizes &sizes,
                                  int man_acc, int exp_acc,
                                  int man_mul, int exp_mul,
                                  int man_div, int exp_div,
                                  int man_sqrt, int exp_sqrt,
                                  bool saturate,
                                  SubnormalsMode subnormals)
{
  int bias_acc = (1 << (exp_acc - 1)) - 1;
  int bias_mul = (1 << (exp_mul - 1)) - 1;
  int bias_div = (1 << (exp_div - 1)) - 1;
  int bias_sqrt = (1 << (exp_sqrt - 1)) - 1;
  layernorm_forward(
      input, weight, bias, output, mean, rstd, eps, sizes,
      [man_acc, exp_acc, bias_acc, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_acc, exp_acc, bias_acc, saturate, subnormals); },
      [man_mul, exp_mul, bias_mul, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_mul, exp_mul, bias_mul, saturate, subnormals); },
      [man_div, exp_div, bias_div, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_div, exp_div, bias_div, saturate, subnormals); },
      [man_sqrt, exp_sqrt, bias_sqrt, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_sqrt, exp_sqrt, bias_sqrt, saturate, subnormals); });
}

void layernorm_backward_fp_nearest(float *input, float *grad_output,
                                   float *weight, float *bias,
                                   float *mean, float *rstd,
                                   float *grad_input, float *grad_gamma, float *grad_beta,
                                   const DimSizes &sizes,
                                   int man_acc, int exp_acc,
                                   int man_mul, int exp_mul,
                                   int man_div, int exp_div,
                                   bool saturate,
                                   SubnormalsMode subnormals)
{
  // creating xhat_gradient, an array of all 0s for backward pass
  // xhat_gradient is an output from the first pass of the backward
  // used again as an input to the second pass of the backward
  float *xhat_gradient;
  int bias_acc = (1 << (exp_acc - 1)) - 1;
  int bias_mul = (1 << (exp_mul - 1)) - 1;
  int bias_div = (1 << (exp_div - 1)) - 1;
  cudaMalloc(&xhat_gradient, sizeof(float) * sizes.outer * sizes.inner * sizes.channel);
  layernorm_backward(
      input, grad_output, weight, bias, mean, rstd, grad_input, grad_gamma, grad_beta, xhat_gradient, sizes,
      [man_acc, exp_acc, bias_acc, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_acc, exp_acc, bias_acc, saturate, subnormals); },
      [man_mul, exp_mul, bias_mul, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_mul, exp_mul, bias_mul, saturate, subnormals); },
      [man_div, exp_div, bias_div, saturate, subnormals] __device__(float x)
      { return cast_fp_nearest_even(x, man_div, exp_div, bias_div, saturate, subnormals); });
  cudaFree(xhat_gradient);
}