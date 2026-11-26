// fixedpoint_pointwise and fixedpoint_pwl on the CPU (cpu/pointwise_kernel.h
// holds the loops): the step, then fixedpoint_quant's cast of the format.

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
    int wl, fl, spec;
    bool is_signed, symmetric;
    RoundMode rm;

    template <class C>
    void operator()(const C &c, uint64_t seed) const
    {
      using T = carrier_t<typename C::scalar_type>;
      auto make = [&](auto m)
      {
        return mptorch::pointwise::make_fixedpoint_round<T, decltype(m)::value>(
            wl, fl, is_signed, symmetric, spec);
      };
      if (is_signed)
        pw::run_mode<true>(c, rm, seed, make);
      else
        pw::run_mode<false>(c, rm, seed, make);
    }
  };

  // The quantizer's format checks, then the canonical round mode.
  Launch checked(const char *name, const Tensor &x, int64_t wl, int64_t fl, int64_t prng_bits,
                 bool is_signed, bool symmetric, int64_t round_mode)
  {
    mptorch::check_quantizer_format(mptorch::fixedpoint_format(wl - (is_signed ? 1 : 0), fl),
                                    mptorch::carrier_bits(x.scalar_type() == at::kDouble),
                                    prng_bits, round_mode, name);
    const RoundMode rm = pw::canonical(round_mode, prng_bits, name);
    return Launch{static_cast<int>(wl), static_cast<int>(fl), static_cast<int>(prng_bits),
                  is_signed, symmetric, rm};
  }
} // namespace

Tensor fixedpoint_pointwise_cpu(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                                int64_t op, int64_t carrier_rounding, int64_t wl, int64_t fl,
                                int64_t prng_bits, bool is_signed, bool symmetric,
                                int64_t round_mode)
{
  constexpr const char *name = "fixedpoint_pointwise";
  const Launch l = checked(name, a, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return pw::run_pointwise(name, a, b, scalar, op, carrier_rounding, l.rm, l);
}

Tensor fixedpoint_pwl_cpu(Tensor x, Tensor entries, double lo, double hi, int64_t tails,
                          bool rsqrt, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed,
                          bool symmetric, int64_t round_mode)
{
  constexpr const char *name = "fixedpoint_pwl";
  const Launch l = checked(name, x, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return pw::run_pwl(name, x, entries, lo, hi, tails, rsqrt, l.rm, l);
}
