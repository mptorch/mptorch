#pragma once

#include "modes.h"

float cast_binaryK_nearest_even(float origin_float, int man_bits, int exp_bits,
                                int bias, bool is_signed,
                                SaturationMode saturation_mode,
                                SubnormalsMode subnormals);

float cast_binaryK_nearest_away(float origin_float, int man_bits, int exp_bits,
                                int bias, bool is_signed,
                                SaturationMode saturation_mode,
                                SubnormalsMode subnormals);

float cast_binaryK_up(float origin_float, int man_bits, int exp_bits, int bias,
                      bool is_signed, SaturationMode saturation_mode,
                      SubnormalsMode subnormals);

float cast_binaryK_down(float origin_float, int man_bits, int exp_bits,
                        int bias, bool is_signed,
                        SaturationMode saturation_mode,
                        SubnormalsMode subnormals);

float cast_binaryK_zero(float origin_float, int man_bits, int exp_bits,
                        int bias, bool is_signed,
                        SaturationMode saturation_mode,
                        SubnormalsMode subnormals);

float cast_binaryK_stochastic(float origin_float, int man_bits, int exp_bits,
                              int rand_bits, int bias, bool is_signed,
                              SaturationMode saturation_mode,
                              SubnormalsMode subnormals);

void binaryK_kernel(
    float *a, float *o, int size, int K, int P, int bias, int prng_bits, bool is_signed,
    RoundMode round_mode = RoundMode::RNE,
    SaturationMode saturation_mode = SaturationMode::OVF_INF,
    SubnormalsMode subnormals_mode = SubnormalsMode::SUBNORMALS);