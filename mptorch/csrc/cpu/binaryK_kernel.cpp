#include "../common/cast_binaryK.h"
#include "../common/modes.h"
#include "../common/dispatch.h"
#include "utils.h"
#include <ATen/ops/empty_like.h>

using namespace at;

namespace
{

  // One body for all six deterministic round modes. Three details in the
  // signature are each worth a measurement (4M f32 elements, e4m3, RNE, one
  // thread; see dev/gemm_perf_audit.md finding C4).
  //
  // `Cast` is the round mode's cast wrapped in a stateless lambda, so every
  // arm of the switch below instantiates this on a distinct closure type and
  // the cast inlines into quant_kernel's loop body. Passing it instead as a
  // std::function<scalar_t(scalar_t)>, as this once did, costs an indirect
  // call per element and a heap allocation for the closure. A function
  // pointer as a non-type template parameter reads more neatly than a
  // closure, but taking a cast's address leaves the compiler free to call an
  // out-of-line copy instead, and measured 18% slower.
  //
  // `IsSigned` is a template parameter rather than an argument because every
  // cast opens with `if (origin_float < 0.0f && !is_signed) return 0.0f;`.
  // As a runtime flag that is a data-dependent early return in the middle of
  // the loop body; as a constant it folds away entirely on the signed path.
  // 52.7 ms runtime vs 20.6 ms constant, for identical arithmetic. The
  // two-way split costs one branch per call and a second instantiation.
  //
  // `p` is a BinaryKParams, so the casts taken here are the precomputed-
  // parameter overloads rather than the (man_bits, exp_bits, bias, ...) ones
  // this used to call. The latter re-derive the rounding masks and the
  // clipping range from those integers on *every element*;
  // make_binaryK_params does it once per tensor.
  //
  // All three want the cast inlined into the loop, which the translation
  // unit is too big for at GCC's default --param inline-unit-growth. See
  // setup.py.
  //
  // A float64 tensor rounds in binary64: the carrier is carrier_t<scalar_t>,
  // float for every other dtype, and the arms' closures take it generically.
  template <typename scalar_t, bool IsSigned, class Cast>
  void binaryK_run(const scalar_t *a, scalar_t *o, int64_t size,
                   SubnormalsMode subnormals_mode, const BinaryKParamsT<carrier_t<scalar_t>> &p, Cast cast)
  {
    quant_kernel(a, o, size,
                 [=](scalar_t x) -> scalar_t
                 {
                   return static_cast<scalar_t>(
                       cast(static_cast<carrier_t<scalar_t>>(x), IsSigned, subnormals_mode, p));
                 });
  }

  template <typename scalar_t, bool IsSigned>
  void binaryK_kernel_impl(const scalar_t *a, scalar_t *o, int64_t size, int K, int P,
                           int bias, RoundMode round_mode,
                           SaturationMode saturation_mode,
                           SubnormalsMode subnormals_mode)
  {
    const int man_bits = P - 1;
    const int exp_bits = IsSigned ? K - P : K - P + 1;
    const BinaryKParamsT<carrier_t<scalar_t>> p = make_binaryK_params<carrier_t<scalar_t>>(
        man_bits, exp_bits, bias, IsSigned, saturation_mode,
        subnormals_mode == SubnormalsMode::EXTENDED_NORMALS);

    switch (round_mode)
    {
    case RoundMode::RNE:
      binaryK_run<scalar_t, IsSigned>(
          a, o, size, subnormals_mode, p,
          [](auto v, bool sg, SubnormalsMode sm, const auto &q)
          { return cast_binaryK_nearest_even(v, sg, sm, q); });
      break;

    case RoundMode::RNA:
      binaryK_run<scalar_t, IsSigned>(
          a, o, size, subnormals_mode, p,
          [](auto v, bool sg, SubnormalsMode sm, const auto &q)
          { return cast_binaryK_nearest_away(v, sg, sm, q); });
      break;

    case RoundMode::RU:
      binaryK_run<scalar_t, IsSigned>(
          a, o, size, subnormals_mode, p,
          [](auto v, bool sg, SubnormalsMode sm, const auto &q)
          { return cast_binaryK_up(v, sg, sm, q); });
      break;

    case RoundMode::RD:
      binaryK_run<scalar_t, IsSigned>(
          a, o, size, subnormals_mode, p,
          [](auto v, bool sg, SubnormalsMode sm, const auto &q)
          { return cast_binaryK_down(v, sg, sm, q); });
      break;

    case RoundMode::RZ:
      binaryK_run<scalar_t, IsSigned>(
          a, o, size, subnormals_mode, p,
          [](auto v, bool sg, SubnormalsMode sm, const auto &q)
          { return cast_binaryK_zero(v, sg, sm, q); });
      break;

    default: // RO
      binaryK_run<scalar_t, IsSigned>(
          a, o, size, subnormals_mode, p,
          [](auto v, bool sg, SubnormalsMode sm, const auto &q)
          { return cast_binaryK_odd(v, sg, sm, q); });
      break;
    }
  }

  template <typename scalar_t, bool IsSigned>
  void binaryK_kernel_sr_impl(const scalar_t *a, scalar_t *o, int64_t size,
                              int K, int P, int bias, int prng_bits, uint64_t seed,
                              SaturationMode saturation_mode,
                              SubnormalsMode subnormals_mode)
  {
    const int man_bits = P - 1;
    const int exp_bits = IsSigned ? K - P : K - P + 1;
    const BinaryKParamsT<carrier_t<scalar_t>> p = make_binaryK_params<carrier_t<scalar_t>>(
        man_bits, exp_bits, bias, IsSigned, saturation_mode,
        subnormals_mode == SubnormalsMode::EXTENDED_NORMALS);

    quant_kernel_sr(a, o, size, seed,
                    [=](scalar_t x, typename FloatTraits<carrier_t<scalar_t>>::word_t rv) -> scalar_t
                    {
                      return static_cast<scalar_t>(cast_binaryK_stochastic(
                          static_cast<carrier_t<scalar_t>>(x), rv, prng_bits, IsSigned,
                          subnormals_mode, p));
                    });
  }

} // namespace

Tensor binaryK_quantize_cpu(Tensor a, int64_t K, int64_t P, int64_t bias,
                            int64_t prng_bits, bool is_signed,
                            int64_t round_mode, int64_t saturation_mode,
                            int64_t subnormals_mode)
{
  // data_ptr() walks storage linearly, so a non-contiguous input would be
  // read in the wrong order. mptorch/quant/ops.py already calls .contiguous(),
  // but a direct torch.ops.mptorch.binaryK_quant call need not.
  auto a_c = a.contiguous();
  auto o = empty_like(a_c);
  const int64_t size = a_c.numel(); // int would truncate past 2^31 elements
  RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
  SubnormalsMode subnormals_mode_ =
      static_cast<SubnormalsMode>(subnormals_mode);
  SaturationMode saturation_mode_ =
      static_cast<SaturationMode>(saturation_mode);

  const int K_ = static_cast<int>(K);
  const int P_ = static_cast<int>(P);
  const int bias_ = static_cast<int>(bias);
  const int prng_bits_ = static_cast<int>(prng_bits);

  if (round_mode_ != RoundMode::SR)
  {
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "binaryK_quantize_cpu", [&]
                                    {
      const scalar_t *p_a = a_c.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      if (is_signed)
        binaryK_kernel_impl<scalar_t, true>(p_a, p_o, size, K_, P_, bias_, round_mode_,
                       saturation_mode_, subnormals_mode_);
      else
        binaryK_kernel_impl<scalar_t, false>(p_a, p_o, size, K_, P_, bias_, round_mode_,
                       saturation_mode_, subnormals_mode_); });
  }
  else
  {
    const uint64_t seed = draw_cpu_seed();
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "binaryK_quantize_cpu_sr", [&]
                                    {
      const scalar_t *p_a = a_c.data_ptr<scalar_t>();
      scalar_t *p_o = o.data_ptr<scalar_t>();
      if (is_signed)
        binaryK_kernel_sr_impl<scalar_t, true>(p_a, p_o, size, K_, P_, bias_, prng_bits_, seed,
                       saturation_mode_, subnormals_mode_);
      else
        binaryK_kernel_sr_impl<scalar_t, false>(p_a, p_o, size, K_, P_, bias_, prng_bits_, seed,
                       saturation_mode_, subnormals_mode_); });
  }

  return o;
}
