#include "quant_kernel.h"
#include "sim_helper.cpp"
#include "bit_helper.cpp"
#include <cstdint>
#include <cmath>

void block_kernel_stochastic(float* a, int* r, float* o, int size, float* max_entry, int man_bits) {
    for (int index = 0; index < size; ++index) {
        uint32_t max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
        uint32_t max_exp = max_entry_bits << 1 >> 24 << 23;
        float base_float = 6 * BITS_TO_FLOAT(&max_exp);

        float target_rebase = a[index] + base_float;
        uint32_t target_bits = FLOAT_TO_BITS(&target_rebase);
        uint32_t rand_prob = (uint32_t)r[index];
        uint32_t quantized = round_bitwise_stochastic(target_bits, rand_prob, man_bits);
        float quantize_float = BITS_TO_FLOAT(&quantized) - base_float;

        uint32_t quantize_bits = FLOAT_TO_BITS(&quantize_float);
        uint32_t clip_quantize = clip_max_exponent(man_bits - 2, max_exp, quantize_bits);
        quantize_float = BITS_TO_FLOAT(&clip_quantize);
        o[index] = quantize_float;
    }
}

void block_kernel_nearest(float* a, float* o, int size, float* max_entry, int man_bits) {
    for (int index = 0; index < size; ++index) {
        uint32_t max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
        uint32_t max_exp = max_entry_bits << 1 >> 24 << 23;
        float base_float = 6 * BITS_TO_FLOAT(&max_exp);

        float target_rebase = a[index] + base_float;
        uint32_t target_bits = FLOAT_TO_BITS(&target_rebase);
        uint32_t quantized = round_bitwise_nearest(target_bits, man_bits);
        float quantize_float = BITS_TO_FLOAT(&quantized) - base_float;

        uint32_t quantize_bits = FLOAT_TO_BITS(&quantize_float);
        uint32_t clip_quantize = clip_max_exponent(man_bits - 2, max_exp, quantize_bits);
        quantize_float = BITS_TO_FLOAT(&clip_quantize);

        o[index] = quantize_float;
    }
}

void block_kernel_sim_stochastic(float* a, float* r, float* o, int size, float* max_entry, int wl) {
    for (int index = 0; index < size; ++index) {
        int exponent = ((int)extract_exponent(&max_entry[index]));
        int sigma = exponent - (wl - 1);
        o[index] = round(a[index], r[index], sigma);
    }
}

void block_kernel_sim_nearest(float* a, float* o, int size, float* max_entry, int wl) {
    for (int index = 0; index < size; ++index) {
        int exponent = ((int)extract_exponent(&max_entry[index]));
        int sigma = exponent - (wl - 1);
        o[index] = nearest_round(a[index], sigma);
    }
}

