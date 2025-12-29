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
  int32_t min_exp = -bias + binades_l;
  int32_t max_exp = (1 << exp_bits) - 1 - bias - binades_h;
  bool subnormal = (target_exp < min_exp);
  bool supnormal = (target_exp > max_exp);

  if (subnormal)
  {
    uint32_t qtarget = round_bitwise_nearest_even(target);

    if (((qtarget << 1 >> 24) - 127) < min_exp - binades_l * (1 << man_bits) + 1) // underflow
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
              ((binades_h == 0) ? ((max_exp + 127u) << 23) | ((1u << 23) - (1u << (23 - man_bits + 1)))
                                : ((max_exp + binades_h * (1 << man_bits) + 126u) << 23));
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
    uint32_t qtarget = round_bitwise_nearest_even(target);
    if (((qtarget << 1 >> 24) - 127) > max_exp + binades_h * (1 << man_bits)) // overflow
    {
      if (saturate)
      {
        qtarget = (target >> 31 << 31) |
                  ((binades_h == 0) ? ((max_exp + 127u) << 23) | ((1u << 23) - (1u << (23 - man_bits + 1)))
                                    : ((max_exp + binades_h * (1 << man_bits) + 126u) << 23));
        return BITS_TO_FLOAT(&qtarget);
      }
      else
      {
        float infty = INFINITY;
        qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
        return BITS_TO_FLOAT(&qtarget);
      }
    }
    ftarget = BITS_TO_FLOAT(&qtarget);
  }
  else
  {
    uint32_t qtarget = round_bitwise_nearest_even(target, man_bits);
    ftarget = BITS_TO_FLOAT(&qtarget);

    // extra clipping if binades_h == 0
    if (binades_h == 0)
    {
      uint32_t qmax = (target >> 31 << 31) | ((max_exp + 127u) << 23) | ((1u << 23) - (1u << (23 - man_bits + 1)));
      float fqmax = BITS_TO_FLOAT(&qmax);
      bool overflow = std::fabs(ftarget) > std::fabs(fqmax);
      if (overflow)
        if (saturate)
          return fqmax;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
    }
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
  int32_t min_exp = -bias + (binades_l - 1);
  int32_t max_exp = (1 << exp_bits) - 1 - bias - binades_h;
  bool subnormal = (target_exp < min_exp);
  bool supnormal = (target_exp > max_exp);
  if (subnormal)
  {
    if (target_exp < min_exp - binades_l * (1 << man_bits)) // underflow
      return 0.0f;
    uint32_t qtarget = round_bitwise_nearest_away(target, 0);

    if (((qtarget << 1 >> 24) - 127) < min_exp - binades_l * (1 << man_bits) + 2)
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

    // extra clipping if binades_h == 0
    if (binades_h == 0)
    {
      uint32_t qmax = (target >> 31 << 31) | ((max_exp + 127) << 23) | ((1 << 23) - (1 << (23 - man_bits + 1)));
      float fqmax = BITS_TO_FLOAT(&qmax);
      bool overflow = std::fabs(ftarget) > std::fabs(fqmax);
      if (overflow)
        if (saturate)
          return fqmax;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
    }
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
  int32_t min_exp = -bias + (binades_l - 1);
  int32_t max_exp = (1 << exp_bits) - 1 - bias - binades_h;
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

    // extra clipping if binades_h == 0
    if (binades_h == 0)
    {
      uint32_t qmax = (target >> 31 << 31) | ((max_exp + 127) << 23) | ((1 << 23) - (1 << (23 - man_bits + 1)));
      float fqmax = BITS_TO_FLOAT(&qmax);
      bool overflow = std::fabs(ftarget) > std::fabs(fqmax);
      if (overflow)
        if (saturate)
          return fqmax;
        else
        {
          float infty = INFINITY;
          uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
          return BITS_TO_FLOAT(&qtarget);
        }
    }
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
  int32_t min_exp = -bias + (binades_l - 1);
  int32_t max_exp = (1 << exp_bits) - 1 - bias - binades_h;
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
  int32_t min_exp = -bias + (binades_l - 1);
  int32_t max_exp = (1 << exp_bits) - 1 - bias - (binades_h - 1);
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

void superfp_kernel(float *a, float *o, int size, int man_bits, int exp_bits, int bias, int prng_bits,
                    int binades_l, int binades_h, bool saturate, RoundMode round_mode)
{
  std::function<float(float)> quantizer;

  switch (round_mode)
  {
  case RoundMode::RNE:
    quantizer = [man_bits, exp_bits, bias, binades_l, binades_h, saturate](float x)
    {
      return cast_superfp_nearest_even(x, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
    };
    break;

  case RoundMode::RNA:
    quantizer = [man_bits, exp_bits, bias, binades_l, binades_h, saturate](float x)
    {
      return cast_superfp_nearest_away(x, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
    };
    break;

  case RoundMode::RU:
    quantizer = [man_bits, exp_bits, bias, binades_l, binades_h, saturate](float x)
    {
      return cast_superfp_up(x, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
    };
    break;

  case RoundMode::RD:
    quantizer = [man_bits, exp_bits, bias, binades_l, binades_h, saturate](float x)
    {
      return cast_superfp_down(x, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
    };
    break;

  case RoundMode::RZ:
    quantizer = [man_bits, exp_bits, bias, binades_l, binades_h, saturate](float x)
    {
      return cast_superfp_zero(x, man_bits, exp_bits, bias, binades_l, binades_h, saturate);
    };
    break;

  default:
    quantizer = [man_bits, exp_bits, bias, prng_bits, binades_l, binades_h, saturate](float x)
    {
      return cast_superfp_stochastic(x, man_bits, exp_bits, prng_bits, bias, binades_l, binades_h, saturate);
    };
    break;
  }

  quant_kernel(a, o, size, quantizer);
}