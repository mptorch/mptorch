#pragma once

#include "modes.h"

void binaryK_kernel_nearest_even(
    float *a, float *o, int size, int K, int P, int bias, bool is_signed,
    SaturationMode saturation_mode,
    SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_nearest_away(
    float *a, float *o, int size, int K, int P, int bias, bool is_signed,
    SaturationMode saturation_mode,
    SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_up(float *a, float *o, int size, int K, int P, int bias,
                       bool is_signed, SaturationMode saturation_mode,
                       SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_down(
    float *a, float *o, int size, int K, int P, int bias, bool is_signed,
    SaturationMode saturation_mode,
    SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_zero(
    float *a, float *o, int size, int K, int P, int bias, bool is_signed,
    SaturationMode saturation_mode,
    SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel_stochastic(
    float *a, float *o, int size, int K, int P, int bias, int prng_bits,
    bool is_signed, SaturationMode saturation_mode,
    SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel(
    float *a, float *o, int size, int K, int P, int bias, bool is_signed,
    RoundMode round_mode = RoundMode::RNE,
    SaturationMode saturation_mode = SaturationMode::OVF_INF,
    SubnormalsMode subnormals_mode = SubnormalsMode::SUBNORMALS,
    int prng_bits = 0);