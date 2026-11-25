// The Meta kernels of the three scaled quantizers (common/scaled_quant_host.h),
// which meta_ops.cpp registers: their elementwise twins' format checks, then
// the scale operands', written on symbolic sizes, and the result the CPU op
// allocates, with nothing computed and no RNG state drawn. A file of their
// own, so that meta_ops.cpp's other kernels are what they were.

#include "../common/scaled_quant_host.h"
#include "../common/stochastic_host.h"
#include "meta_ops.h"
#include <ATen/ops/empty_like.h>

namespace
{
  // The format checks the CPU entry points make (check_quantizer_format,
  // common/stochastic_host.h), worded as the CPU words them.
  void check_format(const Tensor &a, const mptorch::Format &f, int64_t prng_bits, int64_t round_mode,
                    const char *op)
  {
    mptorch::check_quantizer_format(f, mptorch::carrier_bits(a.scalar_type() == at::kDouble), prng_bits,
                                    round_mode, op);
  }
} // namespace

Tensor binaryK_quantize_scaled_meta(Tensor a, Tensor scale, int64_t block_size, int64_t K, int64_t P,
                                    int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                                    int64_t saturation_mode, int64_t subnormals_mode)
{
  check_format(a, mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode),
               prng_bits, round_mode, "binaryK_quant_scaled");
  mptorch::check_scaled_operands(a, scale, block_size, "binaryK_quant_scaled");
  return at::empty_like(a.contiguous());
}

Tensor superfp_quantize_scaled_meta(Tensor a, Tensor scale, int64_t block_size, int64_t man_bits,
                                    int64_t exp_bits, int64_t normal_binades, int64_t bias,
                                    int64_t prng_bits, bool, int64_t round_mode, int64_t saturation_mode)
{
  check_format(a, mptorch::superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode),
               prng_bits, round_mode, "superfp_quant_scaled");
  mptorch::check_scaled_operands(a, scale, block_size, "superfp_quant_scaled");
  return at::empty_like(a.contiguous());
}

Tensor fixedpoint_quantize_scaled_meta(Tensor a, Tensor scale, int64_t block_size, int64_t wl, int64_t fl,
                                       int64_t prng_bits, bool is_signed, bool, int64_t round_mode)
{
  check_format(a, mptorch::fixedpoint_format(wl - (is_signed ? 1 : 0), fl), prng_bits, round_mode,
               "fixedpoint_quant_scaled");
  mptorch::check_scaled_operands(a, scale, block_size, "fixedpoint_quant_scaled");
  return at::empty_like(a.contiguous());
}
