// The Meta kernels of the fused pointwise and table ops (common/pointwise.h), which
// meta_ops.cpp registers: their elementwise twins' format checks, then the
// operands' (common/pointwise_host.h), written on symbolic sizes, and the
// result the CPU op allocates, with nothing computed and no RNG state drawn.
// A file of their own, so that meta_ops.cpp's other kernels are what they
// were.

#include "../common/pointwise_host.h"
#include "../common/stochastic_host.h"
#include "meta_ops.h"

namespace
{
  Tensor pointwise_meta(const Tensor &a, const std::optional<Tensor> &b,
                        std::optional<double> scalar, int64_t op, int64_t carrier_rounding,
                        const mptorch::Format &f, int64_t prng_bits, int64_t round_mode,
                        const char *name)
  {
    mptorch::check_quantizer_format(f, mptorch::carrier_bits(a.scalar_type() == at::kDouble),
                                    prng_bits, round_mode, name);
    mptorch::pointwise::check_pointwise(a, b, scalar, op, carrier_rounding, name);
    return mptorch::pointwise::pointwise_result(a, b);
  }

  Tensor pwl_meta(const Tensor &x, const Tensor &entries, double lo, double hi, int64_t tails,
                  const mptorch::Format &f, int64_t prng_bits, int64_t round_mode, const char *name)
  {
    mptorch::check_quantizer_format(f, mptorch::carrier_bits(x.scalar_type() == at::kDouble),
                                    prng_bits, round_mode, name);
    mptorch::pointwise::check_pwl(x, entries, lo, hi, tails, name);
    return mptorch::pointwise::pointwise_result(x, std::nullopt);
  }
} // namespace

Tensor binaryK_pointwise_meta(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                              int64_t op, int64_t carrier_rounding, int64_t K, int64_t P,
                              int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                              int64_t saturation_mode, int64_t subnormals_mode)
{
  return pointwise_meta(
      a, b, scalar, op, carrier_rounding,
      mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode), prng_bits,
      round_mode, "binaryK_pointwise");
}

Tensor superfp_pointwise_meta(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                              int64_t op, int64_t carrier_rounding, int64_t man_bits,
                              int64_t exp_bits, int64_t normal_binades, int64_t bias,
                              int64_t prng_bits, bool, int64_t round_mode,
                              int64_t saturation_mode)
{
  return pointwise_meta(
      a, b, scalar, op, carrier_rounding,
      mptorch::superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode),
      prng_bits, round_mode, "superfp_pointwise");
}

Tensor fixedpoint_pointwise_meta(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                                 int64_t op, int64_t carrier_rounding, int64_t wl, int64_t fl,
                                 int64_t prng_bits, bool is_signed, bool, int64_t round_mode)
{
  return pointwise_meta(a, b, scalar, op, carrier_rounding,
                        mptorch::fixedpoint_format(wl - (is_signed ? 1 : 0), fl), prng_bits,
                        round_mode, "fixedpoint_pointwise");
}

Tensor binaryK_pwl_meta(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool,
                        int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
                        int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
  return pwl_meta(x, entries, lo, hi, tails,
                  mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode),
                  prng_bits, round_mode, "binaryK_pwl");
}

Tensor superfp_pwl_meta(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool,
                        int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias,
                        int64_t prng_bits, bool, int64_t round_mode, int64_t saturation_mode)
{
  return pwl_meta(x, entries, lo, hi, tails,
                  mptorch::superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode),
                  prng_bits, round_mode, "superfp_pwl");
}

Tensor fixedpoint_pwl_meta(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool,
                           int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool,
                           int64_t round_mode)
{
  return pwl_meta(x, entries, lo, hi, tails,
                  mptorch::fixedpoint_format(wl - (is_signed ? 1 : 0), fl), prng_bits, round_mode,
                  "fixedpoint_pwl");
}
