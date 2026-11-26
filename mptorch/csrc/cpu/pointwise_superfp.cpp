// superfp_pointwise and superfp_pwl on the CPU (cpu/pointwise_kernel.h holds
// the loops): the step, then superfp_quant's cast of the format.

#include "../common/format_host.h"
#include "../quant_ops.h"
#include "pointwise_kernel.h"

using at::Tensor;

namespace
{
  namespace pw = mptorch_cpu::pointwise;

  // The format's cast in the call's round mode, around either step.
  struct Launch
  {
    int man_bits, exp_bits, normal_binades, bias, spec;
    bool is_signed;
    SaturationMode saturation;
    RoundMode rm;

    template <class C>
    void operator()(const C &c, uint64_t seed) const
    {
      using T = carrier_t<typename C::scalar_type>;
      auto make = [&](auto m)
      {
        return mptorch::pointwise::make_superfp_round<T, decltype(m)::value>(
            man_bits, exp_bits, normal_binades, bias, is_signed, spec, saturation);
      };
      if (is_signed)
        pw::run_mode<true>(c, rm, seed, make);
      else
        pw::run_mode<false>(c, rm, seed, make);
    }
  };

  // The quantizer's format checks, then the canonical round mode.
  Launch checked(const char *name, const Tensor &x, int64_t man_bits, int64_t exp_bits,
                 int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed,
                 int64_t round_mode, int64_t saturation_mode)
  {
    mptorch::check_quantizer_format(
        mptorch::superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode),
        mptorch::carrier_bits(x.scalar_type() == at::kDouble), prng_bits, round_mode, name);
    const RoundMode rm = pw::canonical(round_mode, prng_bits, name);
    return Launch{static_cast<int>(man_bits),       static_cast<int>(exp_bits),
                  static_cast<int>(normal_binades), static_cast<int>(bias),
                  static_cast<int>(prng_bits),      is_signed,
                  static_cast<SaturationMode>(saturation_mode), rm};
  }
} // namespace

Tensor superfp_pointwise_cpu(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                             int64_t op, int64_t carrier_rounding, int64_t man_bits,
                             int64_t exp_bits, int64_t normal_binades, int64_t bias,
                             int64_t prng_bits, bool is_signed, int64_t round_mode,
                             int64_t saturation_mode)
{
  constexpr const char *name = "superfp_pointwise";
  const Launch l = checked(name, a, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                           round_mode, saturation_mode);
  return pw::run_pointwise(name, a, b, scalar, op, carrier_rounding, l.rm, l);
}

Tensor superfp_pwl_cpu(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool rsqrt,
                       int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias,
                       int64_t prng_bits, bool is_signed, int64_t round_mode,
                       int64_t saturation_mode)
{
  constexpr const char *name = "superfp_pwl";
  const Launch l = checked(name, x, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                           round_mode, saturation_mode);
  return pw::run_pwl(name, x, entries, lo, hi, tails, rsqrt, l.rm, l);
}
