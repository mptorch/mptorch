#pragma once

#include "subnormals.h"
#include "binary8.h"
#include "binaryK_kernel.h"
#include <cmath>
#include <cstdint>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

uint32_t extract_exponent(float *a);

uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits);

uint32_t round_bitwise_nearest_even(uint32_t target, int man_bits);

uint32_t round_bitwise_nearest_even(uint32_t target);

uint32_t round_bitwise_nearest_away(uint32_t target, int man_bits);

uint32_t round_bitwise_up(uint32_t target, int man_bits);

uint32_t round_bitwise_down(uint32_t target, int man_bits);

uint32_t clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
                       uint32_t quantized_num, bool saturate);

uint32_t clip_max_exponent(int man_bits, uint32_t max_exponent, uint32_t quantized_num);

uint32_t clip_subnormal_range_exponent(int exp_bits, int man_bits, int bias, uint32_t old_num,
                                       uint32_t quantized_num);

uint32_t clip_subnormal_range_exponent_up(int exp_bits, int man_bits, int bias, uint32_t old_num,
                                          uint32_t quantized_num);

uint32_t clip_normal_range_exponent(int exp_bits, int man_bits, int bias, uint32_t old_num,
                                    uint32_t quantized_num, SaturationMode saturation_mode, bool extended_normals = false);

uint32_t clip_normal_range_exponent(int exp_bits, int man_bits, int bias, uint32_t old_num,
                                    uint32_t quantized_num, bool saturate, bool extended_normals = false);

uint32_t binary8_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, OverflowPolicy overflow_policy, bool subnormal);