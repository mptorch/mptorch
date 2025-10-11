#pragma once
#include "modes.h"

void binaryK_kernel_nearest_even(float *a, float *o, int size, int K, int P, int bias,
                                 bool is_signed, SaturationMode saturation_mode,
                                 SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_nearest_away(float *a, float *o, int size, int K, int P, int bias,
                                 bool is_signed, SaturationMode saturation_mode,
                                 SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_up(float *a, float *o, int size, int K, int P, int bias,
                       bool is_signed, SaturationMode saturation_mode,
                       SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_down(float *a, float *o, int size, int K, int P, int bias,
                         bool is_signed, SaturationMode saturation_mode,
                         SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_zero(float *a, float *o, int size, int K, int P, int bias,
                         bool is_signed, SaturationMode saturation_mode,
                         SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_stochastic(float *a, float *o, int size, int K, int P, int bias, int prng_bits,
                               bool is_signed, SaturationMode saturation_mode,
                               SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);