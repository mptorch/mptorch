#include <cstdint>

uint32_t clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
                           uint32_t quantized_num, bool saturate);

uint32_t clip_max_exponent(int man_bits,
                               uint32_t max_exponent,
                               uint32_t quantized_num);

template <typename T>
T clamp_helper(T a, T min, T max);

template <typename T>
T clamp_mask_helper(T a, T min, T max, uint8_t *mask);

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max);

float round(float a, float r, int sigma);
