#pragma once

#include <ATen/ATen.h>

at::Tensor binaryK_quantize_cuda(
    at::Tensor a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits,
    bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);

at::Tensor binaryK_quantize_cpu(
    at::Tensor a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits,
    bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);