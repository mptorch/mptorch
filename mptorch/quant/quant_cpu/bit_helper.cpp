#include "quant_cpu.h"
#include "stdio.h"

unsigned int clip_exponent(int exp_bits, int man_bits, unsigned int old_num,
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

unsigned int clip_max_exponent(int man_bits, unsigned int max_exponent,
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
