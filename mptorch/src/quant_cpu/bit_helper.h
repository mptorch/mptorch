#pragma once

#include "binary8.h"
#include <cmath>
#include <cstdint>

uint32_t extract_exponent(float *a);

uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits);

uint32_t round_bitwise_nearest_p1(uint32_t target, int man_bits);

uint32_t round_bitwise_nearest(uint32_t target, int man_bits);

uint32_t round_bitwise_up(uint32_t target, int man_bits);

uint32_t round_bitwise_down(uint32_t target, int man_bits);

uint32_t clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
              uint32_t quantized_num, bool saturate);

uint32_t clip_max_exponent(int man_bits, uint32_t max_exponent,  uint32_t quantized_num);

uint32_t clip_exponent_with_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                  uint32_t quantized_num, bool saturate = false);

uint32_t clip_exponent_without_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                    uint32_t quantized_num, bool saturate = false);

uint32_t binary8_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, OverflowPolicy overflow_policy, bool subnormal);