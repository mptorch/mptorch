#pragma once

#include "modes.h"

void binaryK_kernel(
    float *a, float *o, int size, int K, int P, int bias, int prng_bits, bool is_signed,
    RoundMode round_mode = RoundMode::RNE,
    SaturationMode saturation_mode = SaturationMode::OVF_INF,
    SubnormalsMode subnormals_mode = SubnormalsMode::SUBNORMALS);