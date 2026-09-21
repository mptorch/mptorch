#include "../common/cast_binaryK.h"
#include "../common/modes.h"
#include "../common/dispatch.h"
#include "utils.h"
#include <ATen/ops/empty_like.h>

using namespace at;

namespace
{

  // Elementwise binaryK cast of `size` values in one deterministic round
  // mode: o[i] = cast(a[i]) for every i, on ATen's thread pool. Three
  // details of the signature are each worth a measurement (4M float32
  // elements, e4m3, RNE, one thread).
  //
  // `Cast` is the round mode's cast wrapped in a stateless lambda, so every
  // arm of the switch in binaryK_kernel_impl instantiates this on a distinct
  // closure type and the cast inlines into quant_kernel's loop body. A
  // std::function<scalar_t(scalar_t)> would cost an indirect call per
  // element and a heap allocation for the closure. A function pointer as a
  // non-type template parameter reads more neatly than a closure, but taking
  // a cast's address leaves the compiler free to call an out-of-line copy
  // instead, and measured 18% slower.
  //
  // `IsSigned` is a template parameter rather than an argument because every
  // cast opens with `if (x < 0 && !is_signed) return 0;`. As a runtime flag
  // that is a data-dependent early return in the middle of the loop body; as
  // a constant it folds away entirely on the signed path: 20.6 ms against
  // 52.7 ms for identical arithmetic. The two-way split costs one branch per
  // call and a second instantiation.
  //
  // `p` is a BinaryKParams built once per tensor by make_binaryK_params, so
  // the cast reads its rounding masks and clipping range from a struct
  // instead of re-deriving them from (man_bits, exp_bits, bias) on every
  // element.
  //
  // All three want the cast inlined into the loop, which this translation
  // unit is too big for at GCC's default --param inline-unit-growth; setup.py
  // raises the parameter.
  //
  // The cast runs in the carrier, carrier_t<scalar_t>: binary64 for a
  // float64 tensor and binary32 for float32, float16 and bfloat16. The
  // closures are generic in the value type, so one set serves both.
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

  // Deterministic-mode driver. The format's field widths follow P3109's
  // binaryK convention: P - 1 stored mantissa bits, and the exponent takes
  // the K bits left after the mantissa and, in a signed format, the sign.
  // The cast's parameters are built once per tensor, and the round mode is
  // dispatched outside the loop, one closure type per arm.
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

  // Stochastic-rounding driver: the same field derivation and parameter
  // build, with quant_kernel_sr (utils.h) handing each element a random
  // word of the carrier's width, keyed on the element's index so the result
  // does not depend on the thread count. The cast takes prng_bits of that
  // word as the rounding offset.
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

  // The body of both entry points: every element of the contiguous `a`
  // rounded into `o`, which is `a` itself for the in-place op. The drivers
  // (utils.h) compute o[i] from a[i] alone and carry no restrict, so a == o
  // is value-safe. Dispatches once on the storage dtype and once on the
  // sign, and draws one seed only when the round mode is SR.
  void binaryK_quantize_into(const Tensor &a, Tensor &o, int64_t K, int64_t P, int64_t bias,
                             int64_t prng_bits, bool is_signed, int64_t round_mode,
                             int64_t saturation_mode, int64_t subnormals_mode)
  {
    const int64_t size = a.numel(); // int would truncate past 2^31 elements
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
      MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_quantize_cpu", [&]
                                      {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
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
      MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_quantize_cpu_sr", [&]
                                      {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();
        if (is_signed)
          binaryK_kernel_sr_impl<scalar_t, true>(p_a, p_o, size, K_, P_, bias_, prng_bits_, seed,
                         saturation_mode_, subnormals_mode_);
        else
          binaryK_kernel_sr_impl<scalar_t, false>(p_a, p_o, size, K_, P_, bias_, prng_bits_, seed,
                         saturation_mode_, subnormals_mode_); });
    }
  }

} // namespace

// The CPU kernel behind mptorch::binaryK_quant: a new tensor of a's shape and
// dtype, every element rounded to the binaryK format (K, P, bias, is_signed)
// in the given round, saturation and subnormals modes (the int64_t arguments
// are the enum values of common/modes.h).
Tensor binaryK_quantize_cpu(Tensor a, int64_t K, int64_t P, int64_t bias,
                            int64_t prng_bits, bool is_signed,
                            int64_t round_mode, int64_t saturation_mode,
                            int64_t subnormals_mode)
{
  // data_ptr() walks storage linearly, so a non-contiguous input would be
  // read in the wrong order. mptorch/quant/ops.py already calls
  // .contiguous(), but a direct torch.ops.mptorch.binaryK_quant call need
  // not, so the copy is made (or skipped, for a contiguous input) here.
  auto a_c = a.contiguous();
  auto o = empty_like(a_c);
  binaryK_quantize_into(a_c, o, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                        subnormals_mode);
  return o;
}

// The CPU kernel behind mptorch::binaryK_quant_: the same rounding written
// over `a`, with no output allocation. The copy the out-of-place op makes of
// a strided input would defeat it (the kernel would write the copy and leave
// the caller's tensor as it was), so a strided tensor is refused instead.
Tensor &binaryK_quantize_cpu_(Tensor &a, int64_t K, int64_t P, int64_t bias,
                              int64_t prng_bits, bool is_signed,
                              int64_t round_mode, int64_t saturation_mode,
                              int64_t subnormals_mode)
{
  TORCH_CHECK(a.is_contiguous(), "binaryK_quant_ writes its argument in place and needs a "
              "contiguous tensor, got strides ", a.strides(), " for sizes ", a.sizes(),
              ": use binaryK_quant, which copies a strided input");
  binaryK_quantize_into(a, a, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                        subnormals_mode);
  return a;
}
