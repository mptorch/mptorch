// The Meta kernels of the two fixed-point conv ops, in the order of their
// CPU entry points in cpu/custom_conv_fixedpoint_entry.cpp. Each body is the
// CPU entry point's with the driver replaced by its meta_* twin from
// common/fixedpoint_meta_host.h. tests/test_entry_copies.py checks that
// they stay copies. meta_ops.cpp registers them.

#include "../common/fixedpoint_meta_host.h"
#include "meta_ops.h"

using at::Tensor;
using namespace mptorch::gemm;
using namespace mptorch::meta;

// The split mac's pass: p = Q_M(RN_c(a * b)), s = Q_A(s + p), over the
// gathered operands. `integer` runs the integer accumulator where the
// carrier path would give the same bits, a hook for the tests.
Tensor fixedpoint_conv_meta(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                            c10::IntArrayRef stride, c10::IntArrayRef padding,
                            c10::IntArrayRef dilation, int64_t groups, int64_t mul_wl,
                            int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl,
                            int64_t acc_fl, bool acc_is_signed, bool acc_symmetric,
                            int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits,
                            bool wrap, bool integer)
{
  return meta_custom_conv_fixedpoint(
      "custom_conv_fixedpoint",
      pack_fixedpoint_split(mul_wl, mul_fl, mul_is_signed, mul_symmetric, acc_wl, acc_fl,
                            acc_is_signed, acc_symmetric, mul_prng_bits, acc_prng_bits, wrap,
                            integer),
      a, b, conv_pass, out_size, stride, padding, dilation, groups, round_mode);
}

// The fused mac's pass: s = Q_F(RN_c(a * b) + s), one rounding a step.
Tensor fixedpoint_conv_fma_meta(Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size,
                                c10::IntArrayRef stride, c10::IntArrayRef padding,
                                c10::IntArrayRef dilation, int64_t groups, int64_t fma_wl,
                                int64_t fma_fl, bool fma_is_signed, bool fma_symmetric,
                                int64_t round_mode, int64_t fma_prng_bits, bool wrap)
{
  return meta_custom_conv_fixedpoint(
      "custom_conv_fixedpoint_fma",
      pack_fixedpoint_fused(fma_wl, fma_fl, fma_is_signed, fma_symmetric, fma_prng_bits, wrap), a,
      b, conv_pass, out_size, stride, padding, dilation, groups, round_mode);
}
