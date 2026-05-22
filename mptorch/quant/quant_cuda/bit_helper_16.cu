#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define HALF_TO_BITS(x) (*reinterpret_cast<uint16_t *>(&(x)))
#define BITS_TO_HALF(x) (*reinterpret_cast<__half *>(&(x)))

#define BFLOAT16_TO_BITS(x) (*reinterpret_cast<uint16_t *>(&(x)))
#define BITS_TO_BFLOAT16(x) (*reinterpret_cast<__nv_bfloat16 *>(&(x)))

__host__ __device__ __forceinline__ uint16_t round_bitwise_nearest_even_bf16(uint16_t target, int man_bits)
{
    if (man_bits >= 7)
        return target;
    uint16_t mask = (1 << (7 - man_bits)) - 1;
    uint16_t tie = 1 << (6 - man_bits);
    uint16_t add_r = target + tie;
    uint16_t quantized = add_r & ~mask;
    uint16_t is_tie = (target & mask) == tie;
    uint16_t odd = (man_bits == 0) ? 0 : 1;
    return quantized & ~((is_tie & odd) << (7 - man_bits));
}

__host__ __device__ __forceinline__ uint16_t round_bitwise_nearest_even_fp16(uint16_t target, int man_bits)
{
    if (man_bits >= 10)
        return target;
    uint16_t mask = (1 << (10 - man_bits)) - 1;
    uint16_t tie = 1 << (9 - man_bits);
    uint16_t add_r = target + tie;
    uint16_t quantized = add_r & ~mask;
    uint16_t is_tie = (target & mask) == tie;
    uint16_t odd = (man_bits == 0) ? 0 : 1;
    return quantized & ~((is_tie & odd) << (10 - man_bits));
}

__host__ __device__ __forceinline__ uint16_t clip_subnormal_range_exponent_bf16(int exp_bits, int man_bits, int bias,
                                                                           uint16_t old_num, uint16_t quantized_num)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = (quantized_num & 0x7FFF) >> 7;
  int min_exponent_store = -(bias - 1) - man_bits + 127;

  uint16_t old_sign = old_num & 0x8000;
  // underflow or round to smallest non zero subnormal value
  if (quantized_exponent_store < min_exponent_store)
  {
    int offset = (quantized_exponent_store == (min_exponent_store - 1));
    quantized_num += offset * (1u << 7);
    quantized_num = (quantized_num & 0x7FFF) | old_sign;
    quantized_num *= offset;
  }

  return quantized_num;
}

__host__ __device__ __forceinline__ uint16_t clip_normal_range_exponent_bf16(int exp_bits, int man_bits, int bias,
                                                                        uint16_t old_num, uint16_t quantized_num,
                                                                        bool saturate, bool extended_normals = false)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = (quantized_num & 0x7FFF) >> 7;
  int max_exponent_store = bias + 127;
  int min_exponent_store = -(bias - 1) + 127 - extended_normals;

  uint16_t old_sign = old_num & 0x8000;
  // handle overflow
  if (quantized_exponent_store > max_exponent_store)
  {
    if (saturate)
    {
      uint16_t max_man = (((1 << 7) - 1) >> (7 - man_bits)) << (7 - man_bits);
      uint16_t max_num = ((uint16_t)max_exponent_store << 7) | max_man;
      quantized_num = old_sign | max_num;
    }
    else
    {
      quantized_num = old_sign | 0x7F80;
    }
  }
  // handle underflow
  else if (quantized_exponent_store < min_exponent_store)
  {
    uint16_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num & 0x007F) > (1 << 6));
    quantized_num = offset * (min_exponent_store << 7);
    quantized_num |= old_sign;
  }

  return quantized_num;
}

__host__ __device__ __forceinline__ uint16_t clip_subnormal_range_exponent_fp16(int exp_bits, int man_bits, int bias,
                                                                           uint16_t old_num, uint16_t quantized_num)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = (quantized_num & 0x7FFF) >> 10;
  int min_exponent_store = -(bias - 1) - man_bits + 15;

  uint16_t old_sign = old_num & 0x8000;
  // underflow or round to smallest non zero subnormal value
  if (quantized_exponent_store < min_exponent_store)
  {
    int offset = (quantized_exponent_store == (min_exponent_store - 1));
    quantized_num += offset * (1u << 10);
    quantized_num = (quantized_num & 0x7FFF) | old_sign;
    quantized_num *= offset;
  }

  return quantized_num;
}

__host__ __device__ __forceinline__ uint16_t clip_normal_range_exponent_fp16(int exp_bits, int man_bits, int bias,
                                                                        uint16_t old_num, uint16_t quantized_num,
                                                                        bool saturate, bool extended_normals = false)
{
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = (quantized_num & 0x7FFF) >> 10;
  int max_exponent_store = bias + 15;
  int min_exponent_store = -(bias - 1) + 15 - extended_normals;

  uint16_t old_sign = old_num & 0x8000;
  // handle overflow
  if (quantized_exponent_store > max_exponent_store)
  {
    if (saturate)
    {
      uint16_t max_man = (((1 << 10) - 1) >> (10 - man_bits)) << (10 - man_bits);
      uint16_t max_num = ((uint16_t)max_exponent_store << 10) | max_man;
      quantized_num = old_sign | max_num;
    }
    else
    {
      quantized_num = old_sign | 0x7C00;
    }
  }
  // handle underflow
  else if (quantized_exponent_store < min_exponent_store)
  {
    uint16_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num & 0x03FF) > (1 << 9));
    quantized_num = offset * (min_exponent_store << 10);
    quantized_num |= old_sign;
  }

  return quantized_num;
}
