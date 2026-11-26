// The CUDA entry points of the fused pointwise and table ops, which handle the
// tensors, the generator and the stream (cuda/pointwise_*.cu hold the
// kernels). This file is a .cpp inside csrc/cuda/, so a CPU-only build
// skips it.

#include "../common/dispatch.h"
#include "../common/format_host.h"
#include "../common/pointwise_host.h"
#include "../common/stochastic_host.h"
#include "../quant_ops.h"
#include "pointwise_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <mutex>

using at::Tensor;

namespace
{
  namespace pw = mptorch::pointwise;
  using mptorch::pointwise_cuda::Launch;

  RoundMode canonical(int64_t round_mode, int64_t &prng_bits, const char *name)
  {
    int variant;
    round_mode = mptorch::sr_canonical(round_mode, variant);
    if (variant >= 0)
      prng_bits = mptorch::sr_encode(prng_bits, variant, name);
    return static_cast<RoundMode>(round_mode);
  }

  mptorch::GemmDtype dtype_of(at::ScalarType st)
  {
    switch (st)
    {
    case at::kDouble:
      return mptorch::GemmDtype::Double;
    case at::kHalf:
      return mptorch::GemmDtype::Half;
    case at::kBFloat16:
      return mptorch::GemmDtype::BFloat16;
    default:
      return mptorch::GemmDtype::Float;
    }
  }

  // A call's operands as the kernels index them (make_geom; copies of the
  // broadcast operands, contiguous, when the geometry is too wide), the
  // result, and the launch's other fields. An empty result has nothing
  // indexed.
  struct Prepared
  {
    Tensor a, b, o;
    Launch l;
  };

  Prepared prepare(const Tensor &a, const std::optional<Tensor> &b, RoundMode rm)
  {
    Prepared p{a, b.has_value() ? *b : Tensor(), pw::pointwise_result(a, b), Launch{}};
    if (p.o.numel() == 0)
      return p;
    if (!pw::make_geom(p.a, b.has_value() ? &p.b : nullptr, p.o, true, p.l.g))
    {
      p.a = a.expand(p.o.sizes()).contiguous();
      if (b.has_value())
        p.b = b->expand(p.o.sizes()).contiguous();
      pw::make_geom(p.a, b.has_value() ? &p.b : nullptr, p.o, true, p.l.g);
    }
    p.l.a = p.a.data_ptr();
    p.l.b = b.has_value() ? p.b.data_ptr() : nullptr;
    p.l.o = p.o.data_ptr();
    p.l.dt = dtype_of(a.scalar_type());
    p.l.rm = rm;
    p.l.stream = at::cuda::getCurrentCUDAStream();
    return p;
  }

  // Under SR, one Philox block of the default generator's offset, reserved
  // as the quantizers reserve it (cuda/utils.cuh's quant_rng_engine_inputs).
  // An empty call has returned before, without drawing, as the quantizers'
  // CUDA bodies return.
  void draw(Launch &l)
  {
    if (l.rm != RoundMode::SR)
      return;
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    l.rng = gen->philox_cuda_state(1);
  }

  // The body every family's arithmetic entry shares, after its format
  // checks: the operands' checks, the result, and `launch(l)`.
  template <class F>
  Tensor run_pointwise(const char *name, const Tensor &a, const std::optional<Tensor> &b,
                       std::optional<double> scalar, int64_t op, int64_t carrier_rounding,
                       RoundMode rm, F &&launch)
  {
    pw::check_pointwise(a, b, scalar, op, carrier_rounding, name);
    Prepared p = prepare(a, b, rm);
    if (p.o.numel() == 0)
      return p.o;
    p.l.op = static_cast<int32_t>(op);
    p.l.odd = carrier_rounding == static_cast<int64_t>(RoundMode::RO);
    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "pointwise_cuda", [&]
                                 { p.l.y = static_cast<double>(
                                       pw::number_in_carrier<carrier_t<scalar_t>, scalar_t>(scalar, p.l.odd)); });
    draw(p.l);
    launch(p.l);
    return p.o;
  }

  // The body every family's table entry shares, likewise.
  template <class F>
  Tensor run_pwl(const char *name, const Tensor &x, const Tensor &entries, double lo, double hi,
                 int64_t tails, bool rsqrt, RoundMode rm, F &&launch)
  {
    pw::check_pwl(x, entries, lo, hi, tails, name);
    Prepared p = prepare(x, std::nullopt, rm);
    if (p.o.numel() == 0)
      return p.o;
    const Tensor e = entries.contiguous();
    p.l.pwl = true;
    p.l.entries = e.data_ptr();
    p.l.segments = static_cast<int32_t>(e.numel() - 1);
    p.l.lo = lo;
    p.l.hi = hi;
    p.l.tails = static_cast<int32_t>(tails);
    p.l.rsqrt = rsqrt;
    draw(p.l);
    launch(p.l);
    return p.o;
  }

  // Each family's format checks and its launch, shared by its two entries.
  struct BinaryK
  {
    mptorch::pointwise_cuda::BinaryKFormat f;
    RoundMode rm;
    void operator()(const Launch &l) const { mptorch::pointwise_cuda::launch_binaryK(l, f); }
  };
  BinaryK binaryK(const char *name, const Tensor &x, int64_t K, int64_t P, int64_t bias,
                  int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode,
                  int64_t subnormals_mode)
  {
    mptorch::check_quantizer_format(
        mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode),
        mptorch::carrier_bits(x.scalar_type() == at::kDouble), prng_bits, round_mode, name);
    const RoundMode rm = canonical(round_mode, prng_bits, name);
    return BinaryK{{static_cast<int>(K), static_cast<int>(P), static_cast<int>(bias),
                    static_cast<int>(prng_bits), is_signed,
                    static_cast<SaturationMode>(saturation_mode),
                    static_cast<SubnormalsMode>(subnormals_mode)},
                   rm};
  }

  struct Superfp
  {
    mptorch::pointwise_cuda::SuperfpFormat f;
    RoundMode rm;
    void operator()(const Launch &l) const { mptorch::pointwise_cuda::launch_superfp(l, f); }
  };
  Superfp superfp(const char *name, const Tensor &x, int64_t man_bits, int64_t exp_bits,
                  int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed,
                  int64_t round_mode, int64_t saturation_mode)
  {
    mptorch::check_quantizer_format(
        mptorch::superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode),
        mptorch::carrier_bits(x.scalar_type() == at::kDouble), prng_bits, round_mode, name);
    const RoundMode rm = canonical(round_mode, prng_bits, name);
    return Superfp{{static_cast<int>(man_bits), static_cast<int>(exp_bits),
                    static_cast<int>(normal_binades), static_cast<int>(bias),
                    static_cast<int>(prng_bits), is_signed,
                    static_cast<SaturationMode>(saturation_mode)},
                   rm};
  }

  struct FixedPoint
  {
    mptorch::pointwise_cuda::FixedPointFormat f;
    RoundMode rm;
    void operator()(const Launch &l) const { mptorch::pointwise_cuda::launch_fixedpoint(l, f); }
  };
  FixedPoint fixedpoint(const char *name, const Tensor &x, int64_t wl, int64_t fl,
                        int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode)
  {
    mptorch::check_quantizer_format(mptorch::fixedpoint_format(wl - (is_signed ? 1 : 0), fl),
                                    mptorch::carrier_bits(x.scalar_type() == at::kDouble),
                                    prng_bits, round_mode, name);
    const RoundMode rm = canonical(round_mode, prng_bits, name);
    return FixedPoint{{static_cast<int>(wl), static_cast<int>(fl), static_cast<int>(prng_bits),
                       is_signed, symmetric},
                      rm};
  }
} // namespace

Tensor binaryK_pointwise_cuda(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                              int64_t op, int64_t carrier_rounding, int64_t K, int64_t P,
                              int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                              int64_t saturation_mode, int64_t subnormals_mode)
{
  constexpr const char *name = "binaryK_pointwise";
  const auto f = binaryK(name, a, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                         subnormals_mode);
  return run_pointwise(name, a, b, scalar, op, carrier_rounding, f.rm, f);
}

Tensor superfp_pointwise_cuda(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                              int64_t op, int64_t carrier_rounding, int64_t man_bits,
                              int64_t exp_bits, int64_t normal_binades, int64_t bias,
                              int64_t prng_bits, bool is_signed, int64_t round_mode,
                              int64_t saturation_mode)
{
  constexpr const char *name = "superfp_pointwise";
  const auto f = superfp(name, a, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                         round_mode, saturation_mode);
  return run_pointwise(name, a, b, scalar, op, carrier_rounding, f.rm, f);
}

Tensor fixedpoint_pointwise_cuda(Tensor a, std::optional<Tensor> b, std::optional<double> scalar,
                                 int64_t op, int64_t carrier_rounding, int64_t wl, int64_t fl,
                                 int64_t prng_bits, bool is_signed, bool symmetric,
                                 int64_t round_mode)
{
  constexpr const char *name = "fixedpoint_pointwise";
  const auto f = fixedpoint(name, a, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return run_pointwise(name, a, b, scalar, op, carrier_rounding, f.rm, f);
}

Tensor binaryK_pwl_cuda(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool rsqrt,
                        int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
                        int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
  constexpr const char *name = "binaryK_pwl";
  const auto f = binaryK(name, x, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                         subnormals_mode);
  return run_pwl(name, x, entries, lo, hi, tails, rsqrt, f.rm, f);
}

Tensor superfp_pwl_cuda(Tensor x, Tensor entries, double lo, double hi, int64_t tails, bool rsqrt,
                        int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias,
                        int64_t prng_bits, bool is_signed, int64_t round_mode,
                        int64_t saturation_mode)
{
  constexpr const char *name = "superfp_pwl";
  const auto f = superfp(name, x, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                         round_mode, saturation_mode);
  return run_pwl(name, x, entries, lo, hi, tails, rsqrt, f.rm, f);
}

Tensor fixedpoint_pwl_cuda(Tensor x, Tensor entries, double lo, double hi, int64_t tails,
                           bool rsqrt, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed,
                           bool symmetric, int64_t round_mode)
{
  constexpr const char *name = "fixedpoint_pwl";
  const auto f = fixedpoint(name, x, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return run_pwl(name, x, entries, lo, hi, tails, rsqrt, f.rm, f);
}
