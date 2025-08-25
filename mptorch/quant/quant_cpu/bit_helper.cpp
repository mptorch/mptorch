#include "binary8.h"
#include "bit_helper.h"
#include <cmath>
#include <cstdint>
#include <algorithm>

// extracts exponent from a float value
uint32_t extract_exponent(float *a)
{
  uint32_t temp = *(reinterpret_cast<uint32_t *>(a));
  // extract exponent bits (single precision, 1 sign bit, 23 mantissa bits)
  temp = (temp << 1 >> 24);
  // adjust for exponent bias and virtual bit
  return temp - 127 + 1;
}

// stochastic rounding on a binary8 format
uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits)
{
  uint32_t mask = (1 << (23 - man_bits)) - 1;

  // add masked random bits to an unmasked target
  uint32_t add_r = target + (rand_prob & mask);

  // mask out bits on the right hand side of the least significant bit
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

// rounds to nearest, ties to even, for P = 1 special case binary8 format
uint32_t round_bitwise_nearest_p1(uint32_t target, int man_bits)
{
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
  // tie breaking rule offset
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

// rounds to nearest, ties to even
uint32_t round_bitwise_nearest(uint32_t target, int man_bits)
{
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
  // tie breaking rule offset
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  return add_r & ~((1 << std::min<int>((23 - man_bits + offset), 23)) - 1);
}

// rounds up, towards positive infinity
uint32_t round_bitwise_up(uint32_t target, int man_bits)
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
uint32_t round_bitwise_down(uint32_t target, int man_bits)
{
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
  uint32_t sign = target >> 31;
  uint32_t rand_prob = (nexact & sign) << (23 - man_bits);
  uint32_t add_r = target + rand_prob;
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

// clips the exponent of a binary8 float value
uint32_t clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
                       uint32_t quantized_num, bool saturate)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // handle overflow
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
      quantized_num =
          ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
      quantized_num = quantized_num | old_sign;
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
uint32_t clip_max_exponent(int man_bits, uint32_t max_exponent, uint32_t quantized_num)
{
  uint32_t quantized_exponent = quantized_num << 1 >> 24 << 23;
  if (quantized_exponent > max_exponent)
  {
    uint32_t max_man = (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
    uint32_t max_num = max_exponent | max_man;
    uint32_t old_sign = quantized_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}

// clips the exponent of a floating point format with subnormal values
uint32_t clip_exponent_with_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                       uint32_t quantized_num, bool saturate)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 2) - man_bits + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // underflow or round to smallest non zero subnormal value
  if (quantized_exponent_store < min_exponent_store)
  {
    int offset = (quantized_exponent_store == (min_exponent_store - 1));
    quantized_num += offset * (1u << 23);
    quantized_num = quantized_num | old_sign;
    quantized_num = offset * quantized_num;
  }
  return quantized_num;
}

// clips the exponent of a floating point format without subnormal values
uint32_t clip_exponent_without_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                          uint32_t quantized_num, bool saturate)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // handle overflow
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
      quantized_num = ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
      quantized_num = quantized_num | old_sign;
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

// clips the exponent for a specified binary8 format
uint32_t binary8_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, OverflowPolicy overflow_policy, bool subnormal)
{
  if (quantized_num == 0)
  {
    return quantized_num;
  }

  uint32_t man_val = quantized_num & 0x7FFFFF;
  uint32_t old_sign = old_num >> 31 << 31;
  uint32_t max_man;

  int spec_exp = (man_bits == 0) ? 1 : 0;
  // special unsigned exponent for the unsigned case as we want
  // our min to be 1 less due to NaN representation
  int special_unsigned_exp = 0;

  max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);

  // handle special case for signed representation with reals number system
  if (exp_bits + man_bits == 7 && overflow_policy == OverflowPolicy::SATURATE_MAXFLOAT2)
  {
    max_man = ((1u << man_bits) - 1u) << (23 - man_bits);
  }

  if (overflow_policy != OverflowPolicy::SATURATE_MAXFLOAT2)
  {
    if (exp_bits == 8)
    {
      special_unsigned_exp = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
    }
    else if (exp_bits == 7 && man_bits == 1)
    {                           // unsigned and p=2
      special_unsigned_exp = 1; // 1 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 1
      max_man = ((1u << man_bits) - 1u) << (23 - man_bits);
    }
    else if (exp_bits + man_bits == 8)
    {                                                       // unsigned
      max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value is 0xfd = max_exp | max_mantissa - 1
    }
  }

  int quantized_exponent_store = (quantized_num << 1 >> 24);
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127 - special_unsigned_exp;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp;

  if (!subnormal)
  {
    min_exponent_store--;
  }

  // handle overflow
  if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && (man_val > max_man)))
  {
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY)
    {
      return quantized_num = old_sign | 0x7F800000; // INF
    }
    quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
  }

  // handle underflow
  uint32_t min_man = 1u << (23 - man_bits);
  if (quantized_exponent_store < min_exponent_store || (quantized_exponent_store == min_exponent_store && man_val < min_man))
  {
    if (subnormal)
    {
      // handle subnormal values
      int subnormal_shift = min_exponent_store - quantized_exponent_store;
      int min_subnormals_exp = min_exponent_store - man_bits;
      uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
      uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
      if (subnormal_shift <= man_bits)
      {
        // quantized_num stays the same in this case
      }
      else if ((old_num & 0x7FFFFFFF) > middle_num)
      {
        quantized_num = old_sign | min_num;
      }
      else
      {
        quantized_num = 0;
      }
    }
    else
    {
      // handle normalized values
      uint32_t min_num = ((uint32_t)(min_exponent_store << 23) | 1 << (23 - man_bits));
      uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23 | 1 << (23 - man_bits));
      if ((old_num & 0x7FFFFFFF) > middle_num)
      {
        return quantized_num = old_sign | min_num;
      }
      else
      {
        quantized_num = 0;
      }
    }
  }
  return quantized_num;
}