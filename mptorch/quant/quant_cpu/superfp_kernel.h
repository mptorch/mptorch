#pragma once
#include "modes.h"

float cast_superfp_nearest_even(float origin_float, int man_bits, int exp_bits, int bias,
                                int binades_l, int binades_h, bool saturate);

float cast_superfp_nearest_away(float origin_float, int man_bits, int exp_bits, int bias,
                                int binades_l, int binades_h, bool saturate);

float cast_superfp_up(float origin_float, int man_bits, int exp_bits, int bias,
                      int binades_l, int binades_h, bool saturate);

float cast_superfp_down(float origin_float, int man_bits, int exp_bits, int bias,
                        int binades_l, int binades_h, bool saturate);

float cast_superfp_zero(float origin_float, int man_bits, int exp_bits, int bias,
                        int binades_l, int binades_h, bool saturate);

float cast_superfp_stochastic(float origin_float, int man_bits, int exp_bits, int prng_bits, int bias,
                              int binades_l, int binades_h, bool saturate);

void superfp_kernel(float *a, float *o, int man_bits, int exp_bits, int bias, int prng_bits,
                    bool saturate = false, RoundMode round_mode = RoundMode::RNE);