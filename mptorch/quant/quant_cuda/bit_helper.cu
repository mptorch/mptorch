#include "modes.h"
#include <cstdint>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

__host__ __device__ __forceinline__ uint32_t extract_exponent(float *a)
{
  uint32_t temp = *(reinterpret_cast<uint32_t *>(a));
  // extract exponent bits (single precision, 1 sign bit, 23 mantissa bits)
  temp = (temp << 1 >> 24);
  // adjust for exponent bias and virtual bit
  return temp - 127 + 1;
}

// stochastic rounding
__host__ __device__ __forceinline__ uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits)
{ // passing number of random bits as second parameter
  // (all the bits after the least significant bit which is based on prng);
  // target is the original number
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  // adding random bits to target (which is not masked)
  uint32_t add_r = target + (rand_prob & mask);
  // masking out bits on the right hand side of the significant bits (truncating)
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

// rounds to nearest, ties to even
__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest_even(uint32_t target, int man_bits)
{
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
  // tie breaking rule offset
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  int shift_value = man_bits == 0 ? 1 << (23 - man_bits + offset) : 1 << std::min<int>((23 - man_bits + offset), 23);
  return (add_r & ~(shift_value - 1)) + offset * (man_bits == 0) * (machine_eps << 1);
}

__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest_even(uint32_t target)
{
  uint32_t man_val = 0x007FFFFF & target;
  // tie breaking rule
  int midpoint = (man_val == 0x00400000);
  uint32_t add_r = target + 0x00400000;
  int target_exp = (add_r << 1 >> 1 >> 23) - 127;
  return (add_r & ~0x007FFFFF) - 0x00800000 * (target_exp % 2 != 0) * midpoint;
}

// rounds to nearest, ties to away
__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest_away(uint32_t target, int man_bits)
{
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
  // tie breaking rule offset
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  int shift_value = man_bits == 0 ? 1 << (23 - man_bits + offset) : 1 << std::min<int>((23 - man_bits + offset), 23);
  return (add_r & ~(shift_value - 1)) + offset * (man_bits > -1) * (machine_eps << 1);
}

// rounds up, towards positive infinity
__host__ __device__ __forceinline__ uint32_t round_bitwise_up(uint32_t target, int man_bits)
{
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
  uint32_t sign = target >> 31;
  uint32_t rand_prob = (nexact & ~sign) << (23 - man_bits);
  uint32_t add_r = target + rand_prob;
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

// rounds down, towards negative infinity
__host__ __device__ __forceinline__ uint32_t round_bitwise_down(uint32_t target, int man_bits)
{
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
  uint32_t sign = target >> 31;
  uint32_t rand_prob = (nexact & sign) << (23 - man_bits);
  uint32_t add_r = target + rand_prob;
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

__host__ __device__ __forceinline__ uint32_t clip_exponent(
    int exp_bits, int man_bits, uint32_t old_num,
    uint32_t quantized_num, bool saturate)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // saturate or overflow
  if (quantized_exponent_store > max_exponent_store)
  {
    if (saturate)
    {
      uint32_t max_man =
          (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
      uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
      quantized_num = old_sign | max_num;
    }
    else
    {
      quantized_num = old_sign | 0x7f800000;
    }
    // handle underflow
  }
  else if (quantized_exponent_store < min_exponent_store)
  {
    uint32_t min_num = ((uint32_t)min_exponent_store << 23);
    uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
    uint32_t unsigned_quantized_num = quantized_num << 1 >> 1;
    if (unsigned_quantized_num > middle_num)
    {
      uint32_t old_sign = old_num >> 31 << 31;
      quantized_num = old_sign | min_num;
    }
    else
    {
      quantized_num = 0;
    }
  }
  return quantized_num;
}

// clips the max exponent
__host__ __device__ __forceinline__ uint32_t clip_max_exponent(
    int man_bits, uint32_t max_exponent, uint32_t quantized_num)
{
  uint32_t quantized_exponent = quantized_num << 1 >> 24 << 23; // 1 sign bit, 23 mantissa bits
  if (quantized_exponent > max_exponent)
  {
    uint32_t max_man = (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits); // 1 sign bit, 8 exponent bits
    uint32_t max_num = max_exponent | max_man;
    uint32_t old_sign = quantized_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}

// clips the exponent of a floating point format with subnormal values
__host__ __device__ __forceinline__ uint32_t clip_subnormal_range_exponent(int exp_bits, int man_bits, int bias,
                                                                           uint32_t old_num, uint32_t quantized_num)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int min_exponent_store = -(bias - 1) - man_bits + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // underflow or round to smallest non zero subnormal value
  if (quantized_exponent_store < min_exponent_store)
  {
    int offset = (quantized_exponent_store == (min_exponent_store - 1));
    quantized_num += offset * (1u << 23);
    quantized_num |= old_sign;
    quantized_num *= offset;
  }

  return quantized_num;
}

__host__ __device__ __forceinline__ uint32_t clip_subnormal_range_exponent_up(int exp_bits, int man_bits, int bias,
                                                                              uint32_t old_num, uint32_t quantized_num)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int min_exponent_store = -(bias - 1) - man_bits + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // underflow or round to smallest non zero subnormal value
  if (quantized_exponent_store < min_exponent_store)
  {
    quantized_num = min_exponent_store << 23;
    quantized_num |= old_sign;
  }

  return quantized_num;
}

// clips the exponent of a floating point format without subnormal values (binaryK version)
__host__ __device__ __forceinline__ uint32_t clip_normal_range_exponent(int exp_bits, int man_bits, int bias,
                                                                        uint32_t old_num, uint32_t quantized_num,
                                                                        SaturationMode saturation_mode, bool extended_normals = false)
{
  if (quantized_num == 0)
    return quantized_num;

  uint32_t sign = old_num >> 31 << 31;
  if ((quantized_num == 0x7F800000 && saturation_mode != SaturationMode::SAT_FINITE) || (quantized_num > 0x7F800000))
    return sign | quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = (bias - 1) + 126 + (man_bits > 1);
  int min_exponent_store = -(bias - 1) + 127 - extended_normals;
  int finite = (saturation_mode == SaturationMode::SAT_FINITE);

  uint32_t max_man = ((0x007FFFFF >> (23 - man_bits)) - 1 + finite) << (23 - man_bits);
  uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;

  // handle overflow
  if (quantized_exponent_store > max_exponent_store)
  {
    switch (saturation_mode)
    {
    case SaturationMode::SAT_FINITE:
      quantized_num = sign | max_num;
      break;

    case SaturationMode::SAT_PROPAGATE:
      quantized_num = sign | max_num;
      break;

    default:
      quantized_num = sign | 0x7F800000;
      break;
    }
  }
  else if (quantized_exponent_store == max_exponent_store)
  {
    // handle overflow
    if (quantized_num > max_num && saturation_mode == SaturationMode::OVF_INF)
      quantized_num = sign | 0x7F800000;
  }
  // handle underflow
  else if (quantized_exponent_store < min_exponent_store)
  {
    uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
    quantized_num = offset * (min_exponent_store << 23);
    quantized_num |= sign;
  }

  return quantized_num;
}

// clips the exponent of a floating point format without subnormal values (IEEE-754 style floats version)
__host__ __device__ __forceinline__ uint32_t clip_normal_range_exponent(int exp_bits, int man_bits, int bias,
                                                                        uint32_t old_num, uint32_t quantized_num,
                                                                        bool saturate, bool extended_normals = false)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = bias + 127;
  int min_exponent_store = -(bias - 1) + 127 - extended_normals;

  uint32_t old_sign = old_num >> 31 << 31;
  // handle overflow
  if (quantized_exponent_store > max_exponent_store)
  {
    if (saturate)
    {
      uint32_t max_man = (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
      uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
      quantized_num = old_sign | max_num;
    }
    else
    {
      quantized_num = old_sign | 0x7F800000;
    }
  }
  // handle underflow
  else if (quantized_exponent_store < min_exponent_store)
  {
    uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
    quantized_num = offset * (min_exponent_store << 23);
    quantized_num |= old_sign;
  }
  return quantized_num;
}