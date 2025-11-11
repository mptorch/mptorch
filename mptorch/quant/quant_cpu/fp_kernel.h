#pragma once
#include "modes.h"

float cast_fp_nearest_even(float origin_float, int man_bits, int exp_bits,
                           int bias, bool saturate, SubnormalsMode subnormals);

float cast_fp_nearest_away(float origin_float, int man_bits, int exp_bits,
                           int bias, bool saturate, SubnormalsMode subnormals);

float cast_fp_up(float origin_float, int man_bits, int exp_bits, int bias,
                 bool saturate, SubnormalsMode subnormals);

float cast_fp_down(float origin_float, int man_bits, int exp_bits, int bias,
                   bool saturate, SubnormalsMode subnormals);

float cast_fp_zero(float origin_float, int man_bits, int exp_bits, int bias,
                   bool saturate, SubnormalsMode subnormals);

float cast_fp_stochastic(float origin_float, int man_bits, int exp_bits,
                         int bias, bool saturate, SubnormalsMode subnormals);

float cast_fp_stochastic(float origin_float, int man_bits, int exp_bits,
                         int rand_bits, int bias, bool saturate,
                         SubnormalsMode subnormals);