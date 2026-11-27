// The fused pointwise and table ops on an Apple GPU (binaryK_pointwise,
// superfp_pointwise, fixedpoint_pointwise and their _pwl siblings): the CPU
// entries' bodies (cpu/pointwise_{binaryK,superfp,fixedpoint}.cpp and
// cpu/pointwise_kernel.h's run_pointwise and run_pwl), in their order, over
// pointwise_kernel.metal. The format is checked and the round mode made
// canonical as the quantizer's entry does it; then the operands' checks
// (common/pointwise_host.h), the result and the geometry; under SR the seed
// is drawn from the CPU generator before the result's size is looked at, as
// the CPU draws it; an empty result returns before any launch. The cast is
// the MPS quantizer's (quantize_tail.h), so a call returns the CPU op's
// result bit for bit.

#include "../common/dispatch.h"
#include "../common/format_host.h"
#include "../common/pointwise_host.h"
#include "../common/stochastic_host.h"
#include "../cpu/utils.h" // draw_cpu_seed
#include "../quant_ops.h"
#include "metal_runtime.h"
#include "pointwise_params.h"
#include "quantize_tail.h"
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

using at::Tensor;

namespace
{
  namespace pw = mptorch::pointwise;

  static_assert(sizeof(pw::PointwiseGeom) == 240,
                "pointwise_kernel.metal reads PointwiseGeom with this layout");

  // A call's round mode and random bits as the kernel takes them: SR_B and
  // SR_C run as SR, with the variant recorded in prng_bits
  // (stochastic_host.h), as on the CPU.
  RoundMode canonical(int64_t round_mode, int64_t &prng_bits, const char *name)
  {
    int variant;
    round_mode = mptorch::sr_canonical(round_mode, variant);
    if (variant >= 0)
      prng_bits = mptorch::sr_encode(prng_bits, variant, name);
    return static_cast<RoundMode>(round_mode);
  }

  // A family's checked format: its quantize.metal lines, its sign, its
  // canonical round mode and random bits.
  struct Format
  {
    std::string lines;
    bool is_signed;
    RoundMode rm;
    int64_t prng_bits;
  };

  Format binaryK(const char *name, const Tensor &x, int64_t K, int64_t P, int64_t bias,
                 int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode,
                 int64_t subnormals_mode)
  {
    mptorch::check_quantizer_format(
        mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode),
        mptorch::carrier_bits(x.scalar_type() == at::kDouble), prng_bits, round_mode, name);
    const RoundMode rm = canonical(round_mode, prng_bits, name);
    return {
        mptorch_mps::binaryK_format_lines(K, P, bias, is_signed, saturation_mode, subnormals_mode),
        is_signed, rm, prng_bits};
  }

  Format superfp(const char *name, const Tensor &x, int64_t man_bits, int64_t exp_bits,
                 int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed,
                 int64_t round_mode, int64_t saturation_mode)
  {
    mptorch::check_quantizer_format(
        mptorch::superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode),
        mptorch::carrier_bits(x.scalar_type() == at::kDouble), prng_bits, round_mode, name);
    const RoundMode rm = canonical(round_mode, prng_bits, name);
    return {mptorch_mps::superfp_format_lines(man_bits, exp_bits, normal_binades, bias, is_signed,
                                              saturation_mode),
            is_signed, rm, prng_bits};
  }

  Format fixedpoint(const char *name, const Tensor &x, int64_t wl, int64_t fl, int64_t prng_bits,
                    bool is_signed, bool symmetric, int64_t round_mode)
  {
    mptorch::check_quantizer_format(mptorch::fixedpoint_format(wl - (is_signed ? 1 : 0), fl),
                                    mptorch::carrier_bits(x.scalar_type() == at::kDouble),
                                    prng_bits, round_mode, name);
    const RoundMode rm = canonical(round_mode, prng_bits, name);
    return {mptorch_mps::fixedpoint_format_lines(wl, fl, is_signed, symmetric), is_signed, rm,
            prng_bits};
  }

  // A call's tensors as the kernel indexes them: the result, and the
  // operands with the geometry, or copies of their broadcast contiguous when
  // the geometry is too wide for the device (make_geom). Nothing is indexed
  // for an empty result.
  struct Operands
  {
    Tensor a, b, o;
    pw::PointwiseGeom g;
  };

  Operands operands(const Tensor &a, const std::optional<Tensor> &b)
  {
    Operands t{a, b.has_value() ? *b : Tensor(), pw::pointwise_result(a, b), pw::PointwiseGeom{}};
    if (t.o.numel() == 0)
      return t;
    if (!pw::make_geom(t.a, b.has_value() ? &t.b : nullptr, t.o, true, t.g))
    {
      t.a = a.expand(t.o.sizes()).contiguous();
      if (b.has_value())
        t.b = b->expand(t.o.sizes()).contiguous();
      pw::make_geom(t.a, b.has_value() ? &t.b : nullptr, t.o, true, t.g);
    }
    return t;
  }

  // The seed under SR, drawn before the result's size is looked at, as the
  // CPU entries draw it.
  uint64_t seed_for(RoundMode rm)
  {
    return rm == RoundMode::SR ? draw_cpu_seed() : 0;
  }

  // Launches pointwise_kernel.metal over a nonempty call: its tail is the
  // quantizer's, then the index type (32 bits when the last thread's last
  // index, below n + 3, fits) and the step's kind. A missing second operand
  // or table binds the first operand in its place, unread.
  void launch(const char *name, const Operands &t, const Tensor *entries,
              mptorch_mps::PointwiseLaunch q, const Format &f)
  {
    const char *storage = mptorch_mps::metal_storage_type(t.a.scalar_type(), name);
    const int64_t n = t.o.numel();
    q.n = static_cast<uint64_t>(n);
    const bool index32 = n <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) - 3;
    const std::string tail = mptorch_mps::quantize_tail(storage, static_cast<int64_t>(f.rm),
                                                        f.is_signed, f.prng_bits, f.lines) +
                             "using mpt_index_t = " + (index32 ? "uint32_t" : "uint64_t") +
                             ";\n#define MPT_PWL " + (entries ? "1" : "0") + "\n";
    mptorch_mps::launch(mptorch_mps::KernelSource::Pointwise, tail,
                        {t.a, t.g.has_b ? t.b : t.a, t.o, entries ? *entries : t.a,
                         mptorch_mps::KernelArg::params(t.g), mptorch_mps::KernelArg::params(q)},
                        static_cast<uint64_t>((n + 3) / 4));
  }

  // The body of every family's arithmetic entry, after its format checks.
  Tensor run_pointwise(const char *name, const Tensor &a, const std::optional<Tensor> &b,
                       std::optional<double> scalar, int64_t op, int64_t carrier_rounding,
                       const Format &f)
  {
    pw::check_pointwise(a, b, scalar, op, carrier_rounding, name);
    const Operands t = operands(a, b);
    const uint64_t seed = seed_for(f.rm);
    if (t.o.numel() == 0)
      return t.o;
    const bool odd = carrier_rounding == static_cast<int64_t>(RoundMode::RO);
    mptorch_mps::PointwiseLaunch q{};
    q.seed = seed;
    q.op = static_cast<int32_t>(op);
    q.odd = odd ? 1 : 0;
    MPTORCH_DISPATCH_QUANT_TYPES(
        a.scalar_type(), "pointwise_mps",
        [&]
        {
          q.y =
              static_cast<float>(pw::number_in_carrier<carrier_t<scalar_t>, scalar_t>(scalar, odd));
        });
    launch(name, t, nullptr, q, f);
    return t.o;
  }

  // The body of every family's table entry, likewise, with the table's
  // constants in the carrier as the CPU's PwlStep takes them (the scale
  // computed in binary64 first, as torch rounds a number).
  Tensor run_pwl(const char *name, const Tensor &x, const Tensor &entries, double lo, double hi,
                 int64_t tails, bool rsqrt, const Format &f)
  {
    pw::check_pwl(x, entries, lo, hi, tails, name);
    const Operands t = operands(x, std::nullopt);
    const uint64_t seed = seed_for(f.rm);
    if (t.o.numel() == 0)
      return t.o;
    const Tensor e = entries.contiguous();
    const int32_t segments = static_cast<int32_t>(e.numel() - 1);
    mptorch_mps::PointwiseLaunch q{};
    q.seed = seed;
    q.segments = segments;
    q.lo = static_cast<float>(lo);
    q.hi = static_cast<float>(hi);
    q.scale = static_cast<float>(static_cast<double>(segments) / (hi - lo));
    q.tails = static_cast<int32_t>(tails);
    q.rsqrt = rsqrt ? 1 : 0;
    launch(name, t, &e, q, f);
    return t.o;
  }
} // namespace

Tensor binaryK_pointwise_mps(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                             int64_t op, int64_t carrier_rounding, int64_t K, int64_t P,
                             int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                             int64_t saturation_mode, int64_t subnormals_mode)
{
  constexpr const char *name = "binaryK_pointwise";
  const Format f = binaryK(name, a, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                           subnormals_mode);
  return run_pointwise(name, a, b, scalar, op, carrier_rounding, f);
}

Tensor superfp_pointwise_mps(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                             int64_t op, int64_t carrier_rounding, int64_t man_bits,
                             int64_t exp_bits, int64_t normal_binades, int64_t bias,
                             int64_t prng_bits, bool is_signed, int64_t round_mode,
                             int64_t saturation_mode)
{
  constexpr const char *name = "superfp_pointwise";
  const Format f = superfp(name, a, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                           round_mode, saturation_mode);
  return run_pointwise(name, a, b, scalar, op, carrier_rounding, f);
}

Tensor fixedpoint_pointwise_mps(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                                int64_t op, int64_t carrier_rounding, int64_t wl, int64_t fl,
                                int64_t prng_bits, bool is_signed, bool symmetric,
                                int64_t round_mode)
{
  constexpr const char *name = "fixedpoint_pointwise";
  const Format f = fixedpoint(name, a, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return run_pointwise(name, a, b, scalar, op, carrier_rounding, f);
}

Tensor binaryK_pwl_mps(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool rsqrt,
                       int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
                       int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
  constexpr const char *name = "binaryK_pwl";
  const Format f = binaryK(name, x, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                           subnormals_mode);
  return run_pwl(name, x, entries, lo, hi, tails, rsqrt, f);
}

Tensor superfp_pwl_mps(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool rsqrt,
                       int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias,
                       int64_t prng_bits, bool is_signed, int64_t round_mode,
                       int64_t saturation_mode)
{
  constexpr const char *name = "superfp_pwl";
  const Format f = superfp(name, x, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                           round_mode, saturation_mode);
  return run_pwl(name, x, entries, lo, hi, tails, rsqrt, f);
}

Tensor fixedpoint_pwl_mps(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool rsqrt,
                          int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric,
                          int64_t round_mode)
{
  constexpr const char *name = "fixedpoint_pwl";
  const Format f = fixedpoint(name, x, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return run_pwl(name, x, entries, lo, hi, tails, rsqrt, f);
}
