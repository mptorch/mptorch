#include "../common/cast_fixedpoint.h"
#include "../common/modes.h"
#include "../common/dispatch.h"
#include "utils.h"
#include <ATen/ops/empty_like.h>

using namespace at;

namespace
{

  // Elementwise fixed-point cast of `size` values in one deterministic round
  // mode, the fixed-point twin of superfp_kernel.cpp's superfp_run: o[i] =
  // cast(a[i]) on ATen's thread pool. `Cast` is a per-arm closure type
  // rather than a function pointer so the cast inlines into the loop body,
  // `IsSigned` is a template parameter so the unsigned early return folds
  // away on the signed path, and `p` is a FixedPointParams built once per
  // tensor rather than re-derived per element. The cast runs in the carrier,
  // carrier_t<scalar_t>: binary64 for float64, binary32 otherwise.
  template <typename scalar_t, bool IsSigned, class Cast>
  void fixedpoint_run(const scalar_t *a, scalar_t *o, int64_t size,
                      const FixedPointParamsT<carrier_t<scalar_t>> &p, Cast cast)
  {
    quant_kernel(a, o, size,
                 [=](scalar_t x) -> scalar_t
                 {
                   return static_cast<scalar_t>(
                       cast(static_cast<carrier_t<scalar_t>>(x), IsSigned, p));
                 });
  }

  // Deterministic-mode driver: builds the cast's parameters (the step and the
  // two ends of the range) once per tensor and dispatches the round mode
  // outside the loop, one closure type per arm.
  template <typename scalar_t, bool IsSigned>
  void fixedpoint_kernel_impl(const scalar_t *a, scalar_t *o, int64_t size, int wl,
                              int fl, bool symmetric, RoundMode round_mode)
  {
    const FixedPointParamsT<carrier_t<scalar_t>> p = make_fixedpoint_params<carrier_t<scalar_t>>(
        wl, fl, IsSigned, symmetric);

    switch (round_mode)
    {
    case RoundMode::RNE:
      fixedpoint_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_fixedpoint_nearest_even(v, sg, q); });
      break;

    case RoundMode::RNA:
      fixedpoint_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_fixedpoint_nearest_away(v, sg, q); });
      break;

    case RoundMode::RU:
      fixedpoint_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_fixedpoint_up(v, sg, q); });
      break;

    case RoundMode::RD:
      fixedpoint_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_fixedpoint_down(v, sg, q); });
      break;

    case RoundMode::RZ:
      fixedpoint_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_fixedpoint_zero(v, sg, q); });
      break;

    default: // RO
      fixedpoint_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_fixedpoint_odd(v, sg, q); });
      break;
    }
  }

  // Stochastic-rounding driver: the same parameter build, with
  // quant_kernel_sr (utils.h) handing each element a random word of the
  // carrier's width, keyed on the element's index so the result does not
  // depend on the thread count. The cast takes prng_bits of that word as
  // the rounding offset.
  template <typename scalar_t, bool IsSigned>
  void fixedpoint_kernel_sr_impl(const scalar_t *a, scalar_t *o, int64_t size, int wl, int fl,
                                 bool symmetric, int prng_bits, uint64_t seed)
  {
    const FixedPointParamsT<carrier_t<scalar_t>> p = make_fixedpoint_params<carrier_t<scalar_t>>(
        wl, fl, IsSigned, symmetric);

    quant_kernel_sr(a, o, size, seed,
                    [=](scalar_t x, typename FloatTraits<carrier_t<scalar_t>>::word_t rv) -> scalar_t
                    {
                      return static_cast<scalar_t>(cast_fixedpoint_stochastic(
                          static_cast<carrier_t<scalar_t>>(x), rv, prng_bits, IsSigned, p));
                    });
  }

  // The body of both entry points, as binaryK_quantize_into is (binaryK_kernel.cpp):
  // `o` is `a` itself for the in-place op. Dispatches once on the storage
  // dtype and once on the sign, and draws one seed only when the round mode
  // is SR.
  void fixedpoint_quantize_into(const Tensor &a, Tensor &o, int64_t wl, int64_t fl,
                                int64_t prng_bits, bool is_signed, bool symmetric,
                                int64_t round_mode)
  {
    const int64_t size = a.numel(); // int would truncate past 2^31 elements
    RoundMode round_mode_ = static_cast<RoundMode>(round_mode);

    const int wl_ = static_cast<int>(wl);
    const int fl_ = static_cast<int>(fl);
    const int prng_bits_ = static_cast<int>(prng_bits);

    if (round_mode_ != RoundMode::SR)
    {
      MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "fixedpoint_quantize_cpu", [&]
                                      {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();
        if (is_signed)
          fixedpoint_kernel_impl<scalar_t, true>(p_a, p_o, size, wl_, fl_, symmetric, round_mode_);
        else
          fixedpoint_kernel_impl<scalar_t, false>(p_a, p_o, size, wl_, fl_, symmetric, round_mode_); });
    }
    else
    {
      const uint64_t seed = draw_cpu_seed();
      MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "fixedpoint_quantize_cpu_sr", [&]
                                      {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();
        if (is_signed)
          fixedpoint_kernel_sr_impl<scalar_t, true>(p_a, p_o, size, wl_, fl_, symmetric,
                         prng_bits_, seed);
        else
          fixedpoint_kernel_sr_impl<scalar_t, false>(p_a, p_o, size, wl_, fl_, symmetric,
                         prng_bits_, seed); });
    }
  }

} // namespace

// The CPU kernel behind mptorch::fixedpoint_quant: a new tensor of a's shape
// and dtype, every element rounded to the fixed-point format (wl, fl,
// is_signed, symmetric) in the given round mode (the int64_t is the enum
// value of common/modes.h). There is no saturation mode: a fixed-point format
// always saturates, having no infinity.
Tensor fixedpoint_quantize_cpu(Tensor a, int64_t wl, int64_t fl, int64_t prng_bits,
                               bool is_signed, bool symmetric, int64_t round_mode)
{
  // data_ptr() walks storage linearly, so a strided input is copied to a
  // contiguous one first (a no-op for an input that already is).
  auto a_c = a.contiguous();
  auto o = empty_like(a_c);
  fixedpoint_quantize_into(a_c, o, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return o;
}

// The CPU kernel behind mptorch::fixedpoint_quant_: the same rounding written
// over `a`, which must be contiguous, as in binaryK_quantize_cpu_.
Tensor &fixedpoint_quantize_cpu_(Tensor &a, int64_t wl, int64_t fl, int64_t prng_bits,
                                 bool is_signed, bool symmetric, int64_t round_mode)
{
  TORCH_CHECK(a.is_contiguous(), "fixedpoint_quant_ writes its argument in place and needs a "
              "contiguous tensor, got strides ", a.strides(), " for sizes ", a.sizes(),
              ": use fixedpoint_quant, which copies a strided input");
  fixedpoint_quantize_into(a, a, wl, fl, prng_bits, is_signed, symmetric, round_mode);
  return a;
}
