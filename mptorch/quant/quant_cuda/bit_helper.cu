#include "binary8_kernel.h"
#include <cmath>
#include <cstdint>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

__host__ __device__ __forceinline__ uint32_t extract_exponent(float *a) {
  uint32_t temp = *(reinterpret_cast<uint32_t *>(a));
  temp = (temp << 1 >> 24); // single precision, 1 sign bit, 23 mantissa bits
  return temp - 127 + 1;    // exponent offset and virtual bit
}

__host__ __device__ __forceinline__ uint32_t
round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits) { // passing number of random bits as second parameter (all the bits after the least significant bit which is based on prng); target is the original number
  uint32_t mask = (1 << (23 - man_bits)) - 1; 
  uint32_t add_r = target + (rand_prob & mask); // adding random bits to target (which is not masked)
  uint32_t quantized = add_r & ~mask; // masking out bits on the right hand side of the significant bits (truncating)
  return quantized;
}

__host__ __device__ __forceinline__ uint32_t
round_bitwise_nearest_p1(uint32_t target, int man_bits) {
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 1 << (22 - man_bits);
  // tie breaking rule offset
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

__host__ __device__ __forceinline__ uint32_t
round_bitwise_nearest(uint32_t target, int man_bits) {
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 1 << (22 - man_bits);
  // tie breaking rule offset
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  return add_r & ~((1 << std::min((23 - man_bits + offset),23)) - 1);
}

__host__ __device__ __forceinline__ uint32_t round_bitwise_up(uint32_t target, int man_bits) {
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
  uint32_t sign = target >> 31;
  uint32_t rand_prob = (nexact & ~sign) << (23 - man_bits);
  uint32_t add_r = target + rand_prob;
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

__host__ __device__ __forceinline__ uint32_t round_bitwise_down(uint32_t target, int man_bits) {
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
  uint32_t sign = target >> 31;
  uint32_t rand_prob = (nexact & sign) << (23 - man_bits);
  uint32_t add_r = target + rand_prob;
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

__host__ __device__ __forceinline__ uint32_t
clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
              uint32_t quantized_num, bool saturate) {
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // saturate or overflow
  if (quantized_exponent_store > max_exponent_store) {
    if (saturate) {
      uint32_t max_man =
          (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
      uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
      quantized_num = old_sign | max_num;
    } else {
      quantized_num =
          ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
      quantized_num = quantized_num | old_sign;
    }
  } else if (quantized_exponent_store < min_exponent_store) {
    uint32_t min_num = ((uint32_t)min_exponent_store << 23);
    uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
    uint32_t unsigned_quantized_num = quantized_num << 1 >> 1;
    if (unsigned_quantized_num > middle_num) {
      uint32_t old_sign = old_num >> 31 << 31;
      quantized_num = old_sign | min_num;
    } else {
      quantized_num = 0;
    }
  }
  return quantized_num;
}

__host__ __device__ __forceinline__ uint32_t
clip_max_exponent(int man_bits, uint32_t max_exponent,  uint32_t quantized_num) {
  uint32_t quantized_exponent = quantized_num << 1 >> 24 << 23; // 1 sign bit, 23 mantissa bits
  if (quantized_exponent > max_exponent) {
    uint32_t max_man = (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits); // 1 sign bit, 8 exponent bits
    uint32_t max_num = max_exponent | max_man;
    uint32_t old_sign = quantized_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}

__host__ __device__ __forceinline__ uint32_t 
binary8_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, SaturationMode saturation_mode, bool subnormal) {

  if (quantized_num == 0){
    return quantized_num;
  }
  
  uint32_t man_val = quantized_num & 0x7FFFFF;
  uint32_t old_sign = old_num >> 31 << 31;
  uint32_t max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
  
  int spec_exp = (man_bits == 0) ? 1 : 0; // if P = 1
  int special_unsigned_exp = 0;
  
  if(exp_bits == 8 && saturation_mode != SaturationMode::NO_OVERFLOW){ // unsigned and p=1
      special_unsigned_exp = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
  }else if (exp_bits == 7 &&  man_bits == 1 && saturation_mode != SaturationMode::NO_OVERFLOW){ // unsigned and p=2 
      special_unsigned_exp = 1; // 1 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 1 
  }else if(exp_bits + man_bits == 8){ // unsigned
      max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value 0xfd = mACax_exp | max_mantissa - 1 
  }

  // Special because in unsigned we want our min to be 1 less because the space is taken by the Nan
  int quantized_exponent_store = (quantized_num << 1 >> 24);
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127 - special_unsigned_exp; 
  int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp;

  if (saturation_mode == SaturationMode::NO_OVERFLOW) { // Saturate to max without infinity
    max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
  }

  if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && (man_val > max_man))) 
  {
    if (saturation_mode == SaturationMode::OVERFLOWS){ // Overflow to infinity
      return quantized_num = old_sign | 0x7F800000; // INF
    } 
    return quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
  }
  if (quantized_exponent_store < min_exponent_store) {
    if (subnormal) {
        int subnormal_shift = min_exponent_store - quantized_exponent_store;
        int min_subnormals_exp = min_exponent_store - man_bits;
        uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
        uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
      if (subnormal_shift <= man_bits) {
        // quantized_num stays the same in this case
      } else if ((old_num & 0x7FFFFFFF) > middle_num) {
        quantized_num = old_sign | min_num;
      } else {
        quantized_num = 0;
      }
    } 
    if(!subnormal) {
      uint32_t min_num = (uint32_t)min_exponent_store << 23;
      uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
      if ((old_num & 0x7FFFFFFF) > middle_num) {
        quantized_num = old_sign | min_num;
      } else {
        quantized_num = 0;
      }
    }
  }
  return quantized_num;
}