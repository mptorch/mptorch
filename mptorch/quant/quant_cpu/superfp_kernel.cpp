#include "superfp_kernel.h"
#include "bit_helper.h"
#include "mm_kernel.h"
#include <random>
#include <iostream>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

float cast_superfp_nearest_even(float origin_float, int man_bits, int exp_bits, int bias,
                                int binades_l, int binades_h, bool saturate)
{
  int32_t sat = saturate;
  uint32_t target;
  target = FLOAT_TO_BITS(&origin_float);
  float ftarget{0u};

  int32_t target_exp = (target << 1 >> 24) - 127;
  int32_t min_exp = 1 - bias + (binades_l - 1);
  int32_t max_exp = (bias - 2) - (binades_h - 1);
  bool subnormal = (target_exp < min_exp);
  bool supnormal = (target_exp > max_exp);
  if (subnormal)
  {
    if (target_exp < min_exp - binades_l * (1 << man_bits)) // underflow
      return 0.0f;
    uint32_t qtarget = round_bitwise_nearest_even(target);

    if (((qtarget << 1 >> 24) - 127) < min_exp - binades_l * (1 << man_bits) + 1)
      return 0.0f;

    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else if (supnormal)
  {
    if (target_exp == 128)
    { // NaN/inf
      if (saturate)
      {
        if ((target & 0x7FFFFFFF) == 0x7F800000)
        { // inf
          uint32_t qtarget =
              (target >> 31 << 31) |
              ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
          return BITS_TO_FLOAT(&qtarget);
        }
        else
        { // NaN
          return origin_float;
        }
      }
      else
      {
        return origin_float;
      }
    }
    else if (target_exp >= max_exp + binades_h * (1 << man_bits) - 1 + sat)
    { // overflow
      if (saturate)
      {
        uint32_t qtarget =
            (target >> 31 << 31) |
            ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
        return BITS_TO_FLOAT(&qtarget);
      }
      else
      {
        if (((target << 9) == 0u) && (target_exp == max_exp + binades_h * (1 << man_bits) - 1))
          return origin_float;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
      }
    }
    uint32_t qtarget = round_bitwise_nearest_even(target);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else
  {
    uint32_t qtarget = round_bitwise_nearest_even(target, man_bits);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }

  return ftarget;
}

float cast_superfp_nearest_away(float origin_float, int man_bits, int exp_bits, int bias,
                                int binades_l, int binades_h, bool saturate)
{
  int32_t sat = saturate;
  uint32_t target;
  target = FLOAT_TO_BITS(&origin_float);
  float ftarget{0u};

  int32_t target_exp = (target << 1 >> 24) - 127;
  int32_t min_exp = 1 - bias + (binades_l - 1);
  int32_t max_exp = (bias - 2) - (binades_h - 1);
  bool subnormal = (target_exp < min_exp);
  bool supnormal = (target_exp > max_exp);
  if (subnormal)
  {
    if (target_exp < min_exp - binades_l * (1 << man_bits)) // underflow
      return 0.0f;
    uint32_t qtarget = round_bitwise_nearest_away(target, 0);

    if (((qtarget << 1 >> 24) - 127) < min_exp - binades_l * (1 << man_bits) + 1)
      return 0.0f;

    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else if (supnormal)
  {
    if (target_exp == 128)
    { // NaN/inf
      if (saturate)
      {
        if ((target & 0x7FFFFFFF) == 0x7F800000)
        { // inf
          uint32_t qtarget =
              (target >> 31 << 31) |
              ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
          return BITS_TO_FLOAT(&qtarget);
        }
        else
        { // NaN
          return origin_float;
        }
      }
      else
      {
        return origin_float;
      }
    }
    else if (target_exp >= max_exp + binades_h * (1 << man_bits) - 1 + sat)
    { // overflow
      if (saturate)
      {
        uint32_t qtarget =
            (target >> 31 << 31) |
            ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
        return BITS_TO_FLOAT(&qtarget);
      }
      else
      {
        if (((target << 9) == 0u) && (target_exp == max_exp + binades_h * (1 << man_bits) - 1))
          return origin_float;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
      }
    }
    uint32_t qtarget = round_bitwise_nearest_away(target, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else
  {
    uint32_t qtarget = round_bitwise_nearest_away(target, man_bits);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }

  return ftarget;
}

float cast_absolute_up(float origin_float, int man_bits, int exp_bits, int bias,
                       int binades_l, int binades_h, bool saturate)
{
  if (origin_float == 0.0f)
    return 0.0f;

  int32_t sat = saturate;
  uint32_t target;
  target = FLOAT_TO_BITS(&origin_float);
  float ftarget{0u};

  int32_t target_exp = (target << 1 >> 24) - 127;
  int32_t min_exp = 1 - bias + (binades_l - 1);
  int32_t max_exp = (bias - 2) - (binades_h - 1);
  bool subnormal = (target_exp < min_exp);
  bool supnormal = (target_exp > max_exp);
  if (subnormal)
  {
    int32_t min_subnormal_exp = min_exp - binades_l * (1 << man_bits) + 1;
    if (target_exp < min_subnormal_exp)
    { // underflow
      uint32_t qtarget_min = (min_subnormal_exp + 127) << 23;
      return FLOAT_TO_BITS(&qtarget_min);
    }
    uint32_t qtarget = round_bitwise_up(target, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else if (supnormal)
  {
    if (target_exp == 128)
    { // NaN/inf
      if (saturate)
      {
        if ((target & 0x7FFFFFFF) == 0x7F800000)
        { // inf
          uint32_t qtarget = ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
          return BITS_TO_FLOAT(&qtarget);
        }
        else
        { // NaN
          return origin_float;
        }
      }
      else
      {
        return origin_float;
      }
    }
    else if (target_exp >= max_exp + binades_h * (1 << man_bits) - 1 + sat)
    { // overflow
      if (saturate)
      {
        uint32_t qtarget = ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
        return BITS_TO_FLOAT(&qtarget);
      }
      else
      {
        if (((target << 9) == 0u) && (target_exp == max_exp + binades_h * (1 << man_bits) - 1))
          return origin_float;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
      }
    }
    uint32_t qtarget = round_bitwise_up(target, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else
  {
    uint32_t qtarget = round_bitwise_up(target, man_bits);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }

  return ftarget;
}

float cast_absolute_down(float origin_float, int man_bits, int exp_bits, int bias,
                         int binades_l, int binades_h, bool saturate)
{
  if (origin_float == 0.0f)
    return 0.0f;

  int32_t sat = saturate;
  uint32_t target;
  target = FLOAT_TO_BITS(&origin_float);
  float ftarget{0u};

  int32_t target_exp = (target << 1 >> 24) - 127;
  int32_t min_exp = 1 - bias + (binades_l - 1);
  int32_t max_exp = (bias - 2) - (binades_h - 1);
  bool subnormal = (target_exp < min_exp);
  bool supnormal = (target_exp > max_exp);
  if (subnormal)
  {
    int32_t min_subnormal_exp = min_exp - binades_l * (1 << man_bits) + 1;
    if (target_exp < min_subnormal_exp)
    { // underflow
      return 0.0f;
    }
    uint32_t qtarget = round_bitwise_down(target, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else if (supnormal)
  {
    if (target_exp == 128)
    { // NaN/inf
      if (saturate)
      {
        if ((target & 0x7FFFFFFF) == 0x7F800000)
        { // inf
          uint32_t qtarget = ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
          return BITS_TO_FLOAT(&qtarget);
        }
        else
        { // NaN
          return origin_float;
        }
      }
      else
      {
        return origin_float;
      }
    }
    else if (target_exp >= max_exp + binades_h * (1 << man_bits) - 1 + sat)
    { // overflow
      if (saturate)
      {
        uint32_t qtarget = ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
        return BITS_TO_FLOAT(&qtarget);
      }
      else
      {
        if (((target << 9) == 0u) && (target_exp == max_exp + binades_h * (1 << man_bits) - 1))
          return origin_float;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
      }
    }
    uint32_t qtarget = round_bitwise_down(target, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else
  {
    uint32_t qtarget = round_bitwise_down(target, man_bits);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }

  return ftarget;
}

float cast_superfp_up(float origin_float, int man_bits, int exp_bits, int bias,
                      int binades_l, int binades_h, bool saturate)
{
  if (origin_float >= 0.0f)
    return cast_absolute_up(origin_float, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
  else
    return -cast_absolute_down(origin_float, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
}

float cast_superfp_down(float origin_float, int man_bits, int exp_bits, int bias,
                        int binades_l, int binades_h, bool saturate)
{
  if (origin_float >= 0.0f)
    return cast_absolute_down(origin_float, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
  else
    return -cast_absolute_up(origin_float, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
}

float cast_superfp_zero(float origin_float, int man_bits, int exp_bits, int bias,
                        int binades_l, int binades_h, bool saturate)
{
  if (origin_float >= 0.0f)
    return cast_superfp_down(origin_float, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
  else
    return cast_superfp_up(origin_float, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
}

float cast_superfp_stochastic(float origin_float, int man_bits, int exp_bits, int prng_bits, int bias,
                              int binades_l, int binades_h, bool saturate)
{
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  thread_local std::uniform_int_distribution<> dis(0);

  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t rand_prob = (dis(gen)) & mask;
  rand_prob = rand_prob << 9 >> 9;

  int32_t sat = saturate;
  uint32_t target;
  target = FLOAT_TO_BITS(&origin_float);
  float ftarget{0u};

  int32_t target_exp = (target << 1 >> 24) - 127;
  int32_t min_exp = 1 - bias + (binades_l - 1);
  int32_t max_exp = (bias - 2) - (binades_h - 1);
  bool subnormal = (target_exp < min_exp);
  bool supnormal = (target_exp > max_exp);

  if (subnormal)
  {
    rand_prob = ~((1 << (23 - prng_bits)) - 1);
    uint32_t qtarget = round_bitwise_stochastic(target, rand_prob, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else if (supnormal)
  {
    if (target_exp == 128)
    { // NaN/inf
      if (saturate)
      {
        if ((target & 0x7FFFFFFF) == 0x7F800000)
        { // inf
          uint32_t qtarget = ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
          return BITS_TO_FLOAT(&qtarget);
        }
        else
        { // NaN
          return origin_float;
        }
      }
      else
      {
        return origin_float;
      }
    }
    else if (target_exp >= max_exp + binades_h * (1 << man_bits) - 1 + sat)
    { // overflow
      if (saturate)
      {
        uint32_t qtarget = ((max_exp + binades_h * (1 << man_bits) + 127u) << 23);
        return BITS_TO_FLOAT(&qtarget);
      }
      else
      {
        if (((target << 9) == 0u) && (target_exp == max_exp + binades_h * (1 << man_bits) - 1))
          return origin_float;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
      }
    }
    rand_prob = ~((1 << (23 - prng_bits)) - 1);
    uint32_t qtarget = round_bitwise_stochastic(target, rand_prob, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else
  {
    rand_prob = ~((1 << (23 - man_bits - prng_bits)) - 1);
    uint32_t qtarget = round_bitwise_stochastic(target, rand_prob, 0);
    ftarget = BITS_TO_FLOAT(&qtarget);
  }

  return ftarget;
}