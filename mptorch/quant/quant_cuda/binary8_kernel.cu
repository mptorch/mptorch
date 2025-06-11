#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include "binary8_kernel.h"
#include "softmax_kernel.h"
#include "layernorm_kernel.h"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__ float cast_binary8_signed_nearest(float origin_float,
                                                      int P, OverflowPolicy overflow_policy,
                                                      bool subnormals)
{

  const int exp_bits = 8 - P;
  const int man_bits = P - 1;
  const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  const uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  int subnormal_shift = 0;
  if (subnormals)
  {
    const int spec_exp = (P == 1) ? 1 : 0;
    const int max_exp = (1 << (exp_bits - 1)) - 1;
    const int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && (exp_val < min_exp))
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                            : round_bitwise_nearest(uval32, man_bits - subnormal_shift);

  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
  return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_binary8_signed_stochastic(
    float origin_float, int P,
    uint32_t rand_prob, int prng_bits,
    OverflowPolicy overflow_policy, bool subnormals)
{

  const int exp_bits = 8 - P;
  const int man_bits = P - 1;
  const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  const uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  int subnormal_shift = 0;
  if (subnormals)
  {
    int spec_exp = (P == 1) ? 1 : 0;
    int max_exp = (1 << (exp_bits - 1)) - 1;
    int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  rand_prob = rand_prob << 9 >> 9;
  // rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);
  rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

  uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

  return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_binary8_signed_truncate(
    float origin_float, int P,
    OverflowPolicy overflow_policy, bool subnormals)
{

  const int exp_bits = 8 - P;
  const int man_bits = P - 1;
  const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  const uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  int subnormal_shift = 0;
  if (subnormals)
  {
    const int spec_exp = (P == 1) ? 1 : 0;
    const int max_exp = (1 << (exp_bits - 1)) - 1;
    const int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  uint32_t uval8 = uval32 >> (23 - man_bits + subnormal_shift) << (23 - man_bits + subnormal_shift);
  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
  return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_binary8_unsigned_nearest(
    float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals)
{

  if (origin_float < 0)
    return NAN;

  uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  const int exp_bits = 9 - P;
  const int man_bits = P - 1;
  int subnormal_shift = 0;

  if (subnormals)
  {
    const int max_exp = (1 << (exp_bits - 1)) - 1;
    const int min_exp = (P == 1) - max_exp;

    if ((min_exp - exp_val) <= man_bits && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                            : round_bitwise_nearest(uval32, man_bits - subnormal_shift);

  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

  return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_binary8_unsigned_stochastic(
    float origin_float, int P,
    uint32_t rand_prob, int prng_bits,
    OverflowPolicy overflow_policy, bool subnormals)
{

  if (origin_float < 0)
    return NAN;

  uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  const int exp_bits = 8 - P + 1;
  const int man_bits = P - 1;
  int subnormal_shift = 0;

  if (subnormals)
  {
    const int spec_exp = (P == 1) ? 1 : 0;
    const int max_exp = (1 << (exp_bits - 1)) - 1;
    const int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  rand_prob = rand_prob << 9 >> 9;
  // rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);
  rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

  uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

  return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_binary8_unsigned_truncate(
    float origin_float, int P,
    OverflowPolicy overflow_policy, bool subnormals)
{

  if (origin_float < 0)
    return NAN;

  uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  const int exp_bits = 8 - P;
  const int man_bits = P - 1;
  int subnormal_shift = 0;

  if (subnormals)
  {
    const int spec_exp = (P == 1) ? 1 : 0;
    const int max_exp = (1 << (exp_bits - 1)) - 1;
    const int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  uint32_t uval8 = uval32 >> (23 - man_bits + subnormal_shift) << (23 - man_bits + subnormal_shift);
  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
  return BITS_TO_FLOAT(&uval8);
}

__global__ void binary8_signed_kernel_nearest(
    float *__restrict__ a, float *o, int size,
    int P, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    o[idx] = cast_binary8_signed_nearest(a[idx], P, overflow_policy, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_nearest(
    float *__restrict__ a, float *o, int size,
    int P, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    o[idx] = cast_binary8_unsigned_nearest(a[idx], P, overflow_policy, subnormals);
  }
}

__global__ void binary8_signed_kernel_stochastic(
    float *__restrict__ a, int *__restrict__ r, float *o, int size,
    int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    o[idx] = cast_binary8_signed_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_stochastic(
    float *__restrict__ a, int *__restrict__ r, float *o, int size,
    int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    o[idx] = cast_binary8_unsigned_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
  }
}

__global__ void binary8_signed_kernel_truncate(
    float *__restrict__ a, float *o, int size,
    int P, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    o[idx] = cast_binary8_signed_truncate(a[idx], P, overflow_policy, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_truncate(
    float *__restrict__ a, float *o, int size,
    int P, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    o[idx] = cast_binary8_unsigned_truncate(a[idx], P, overflow_policy, subnormals);
  }
}

void softmax_forward_binary8_nearest(float *a, float *o, const DimSizes &sizes,
                                     int P_exp, OverflowPolicy op_exp, bool signed_exp,
                                     int P_off, OverflowPolicy op_off, bool signed_off,
                                     int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                     bool subnormals)
{
  softmax_forward(
      a, o, sizes,
      [subnormals, P_exp, op_exp, signed_exp] __device__(float x)
      {
        if (signed_exp)
        {
          return cast_binary8_signed_nearest(x, P_exp, op_exp, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_exp, op_exp, subnormals);
      },
      [subnormals, P_off, op_off, signed_off] __device__(float x)
      {
        if (signed_off)
        {
          return cast_binary8_signed_nearest(x, P_off, op_off, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_off, op_off, subnormals);
      },
      [subnormals, P_acc, op_acc, signed_acc] __device__(float x)
      {
        if (signed_acc)
        {
          return cast_binary8_signed_nearest(x, P_acc, op_acc, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_acc, op_acc, subnormals);
      });
}

void softmax_lse_forward_binary8_nearest(float *a, float *o, const DimSizes &sizes,
                                         int P_off, OverflowPolicy op_off, bool signed_off,
                                         int P_lse, OverflowPolicy op_lse, bool signed_lse,
                                         bool subnormals)
{
  softmax_lse_forward(
      a, o, sizes,
      [subnormals, P_off, op_off, signed_off] __device__(float x)
      {
        if (signed_off)
        {
          return cast_binary8_signed_nearest(x, P_off, op_off, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_off, op_off, subnormals);
      },
      [subnormals, P_lse, op_lse, signed_lse] __device__(float x)
      {
        if (signed_lse)
        {
          return cast_binary8_signed_nearest(x, P_lse, op_lse, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_lse, op_lse, subnormals);
      });
}

void softmax_backward_binary8_nearest(float *a, float *g, float *o, const DimSizes &sizes,
                                      int P_add, OverflowPolicy op_add, bool signed_add,
                                      int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                      bool subnormals)
{
  softmax_backward(
      a, g, o, sizes,
      [subnormals, P_add, op_add, signed_add] __device__(float x)
      {
        if (signed_add)
        {
          return cast_binary8_signed_nearest(x, P_add, op_add, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_add, op_add, subnormals);
      },
      [subnormals, P_mul, op_mul, signed_mul] __device__(float x)
      {
        if (signed_mul)
        {
          return cast_binary8_signed_nearest(x, P_mul, op_mul, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_mul, op_mul, subnormals);
      });
}

void layernorm_forward_binary8_nearest(float *input, float *weight, float *bias,
                                       float *output, float *mean, float *rstd,
                                       float eps, const DimSizes &sizes,
                                       int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                       int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                       int P_div, OverflowPolicy op_div, bool signed_div,
                                       int P_sqrt, OverflowPolicy op_sqrt, bool signed_sqrt,
                                       bool subnormals)
{
  layernorm_forward(input, weight, bias, output, mean, rstd, eps, sizes, [subnormals, P_acc, op_acc, signed_acc] __device__(float x)
                    {
        if (signed_acc){
            return cast_binary8_signed_nearest(x, P_acc, op_acc, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_acc, op_acc, subnormals); }, [subnormals, P_mul, op_mul, signed_mul] __device__(float x)
                    {
        if (signed_mul){
            return cast_binary8_signed_nearest(x, P_mul, op_mul, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_mul, op_mul, subnormals); }, [subnormals, P_div, op_div, signed_div] __device__(float x)
                    {
        if (signed_div){
            return cast_binary8_signed_nearest(x, P_div, op_div, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_div, op_div, subnormals); }, [subnormals, P_sqrt, op_sqrt, signed_sqrt] __device__(float x)
                    {
        if (signed_sqrt){
            return cast_binary8_signed_nearest(x, P_sqrt, op_sqrt, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_sqrt, op_sqrt, subnormals); });
}

void layernorm_backward_binary8_nearest(float *input, float *grad_output,
                                        float *weight, float *bias,
                                        float *mean, float *rstd,
                                        float *grad_input, float *grad_gamma, float *grad_beta,
                                        const DimSizes &sizes,
                                        int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                        int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                        int P_div, OverflowPolicy op_div, bool signed_div,
                                        bool subnormals)
{
  // creating xhat_gradient, an array of all 0s for backward pass
  // xhat_gradient is an output from the first pass of the backward
  // used again as an input to the second pass of the backward
  float *xhat_gradient;
  cudaMalloc(&xhat_gradient, sizeof(float) * sizes.outer * sizes.inner * sizes.channel);
  layernorm_backward(input, grad_output, weight, bias, mean, rstd, grad_input, grad_gamma, grad_beta, xhat_gradient, sizes, [subnormals, P_acc, op_acc, signed_acc] __device__(float x)
                     {
        if (signed_acc){
            return cast_binary8_signed_nearest(x, P_acc, op_acc, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_acc, op_acc, subnormals); }, [subnormals, P_mul, op_mul, signed_mul] __device__(float x)
                     {
        if (signed_mul){
            return cast_binary8_signed_nearest(x, P_mul, op_mul, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_mul, op_mul, subnormals); }, [subnormals, P_div, op_div, signed_div] __device__(float x)
                     {
        if (signed_div){
            return cast_binary8_signed_nearest(x, P_div, op_div, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_div, op_div, subnormals); });
  cudaFree(xhat_gradient);
}