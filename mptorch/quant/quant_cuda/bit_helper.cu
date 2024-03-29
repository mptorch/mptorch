#include <cmath>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

/* __device__ unsigned int rn_prob[24] = {
    4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768,
    16384,   8192,    4096,    2048,   1024,   512,    256,   128,
    64,      32,      16,      8,      4,      2,      1,     0}; */

__device__ __forceinline__ unsigned int extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int *>(a));
  temp = (temp << 1 >> 24); // single precision, 1 sign bit, 23 mantissa bits
  return temp - 127 + 1;    // exponent offset and virtual bit
}

__device__ __forceinline__ unsigned int
round_bitwise_stochastic(unsigned int target, unsigned int rand_prob,
                         int man_bits) {
  unsigned int mask = (1 << (23 - man_bits)) - 1;
  unsigned int add_r = target + (rand_prob & mask);
  unsigned int quantized = add_r & ~mask;
  return quantized;
}

__device__ __forceinline__ unsigned int
round_bitwise_nearest(unsigned int target, int man_bits) {
  unsigned int mask = (1 << (23 - man_bits)) - 1;
  unsigned int rand_prob = 1 << (23 - man_bits - 1);
  // unsigned int rand_prob = rn_prob[man_bits];
  unsigned int add_r = target + rand_prob;
  unsigned int quantized = add_r & ~mask;
  return quantized;
}

__device__ __forceinline__ unsigned int round_bitwise_up(unsigned int target,
                                                         int man_bits) {
  unsigned int mask = (1 << (23 - man_bits)) - 1;

  unsigned int nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
  unsigned int sign = target >> 31;
  unsigned int rand_prob = (nexact & ~sign) << (23 - man_bits);
  unsigned int add_r = target + rand_prob;
  unsigned int quantized = add_r & ~mask;
  return quantized;
}

__device__ __forceinline__ unsigned int round_bitwise_down(unsigned int target,
                                                           int man_bits) {
  unsigned int mask = (1 << (23 - man_bits)) - 1;

  unsigned int nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
  unsigned int sign = target >> 31;
  unsigned int rand_prob = (nexact & sign) << (23 - man_bits);
  unsigned int add_r = target + rand_prob;
  unsigned int quantized = add_r & ~mask;
  return quantized;
}

__device__ __forceinline__ unsigned int
clip_exponent(int exp_bits, int man_bits, unsigned int old_num,
              unsigned int quantized_num, bool saturate = false) {
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

  unsigned int old_sign = old_num >> 31 << 31;
  // saturate or overflow
  if (quantized_exponent_store > max_exponent_store) {
    if (saturate) {
      unsigned int max_man =
          (unsigned int)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
      unsigned int max_num = ((unsigned int)max_exponent_store << 23) | max_man;
      quantized_num = old_sign | max_num;
    } else {
      quantized_num =
          ((((unsigned int)1 << 31) - 1) ^ (((unsigned int)1 << 23) - 1));
      quantized_num = quantized_num | old_sign;
    }
  } else if (quantized_exponent_store < min_exponent_store) {
    unsigned int min_num = ((unsigned int)min_exponent_store << 23);
    unsigned int middle_num = ((unsigned int)(min_exponent_store - 1) << 23);
    unsigned int unsigned_quantized_num = quantized_num << 1 >> 1;
    if (unsigned_quantized_num > middle_num) {
      unsigned int old_sign = old_num >> 31 << 31;
      quantized_num = old_sign | min_num;
    } else {
      quantized_num = 0;
    }
  }
  return quantized_num;
}

__device__ __forceinline__ unsigned int
clip_max_exponent(int man_bits, unsigned int max_exponent,
                  unsigned int quantized_num) {
  unsigned int quantized_exponent =
      quantized_num << 1 >> 24 << 23; // 1 sign bit, 23 mantissa bits
  if (quantized_exponent > max_exponent) {
    unsigned int max_man =
        (unsigned int)-1 << 9 >> 9 >>
        (23 - man_bits) << (23 - man_bits); // 1 sign bit, 8 exponent bits
    unsigned int max_num = max_exponent | max_man;
    unsigned int old_sign = quantized_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}
