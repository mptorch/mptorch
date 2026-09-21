#include "../common/cast_superfp.h"
#include "../common/modes.h"
#include "../common/dispatch.h"
#include "utils.h"
#include <ATen/ops/empty_like.h>

using namespace at;

namespace
{

  // Elementwise superfp cast of `size` values in one deterministic round
  // mode, the superfp twin of binaryK_kernel.cpp's binaryK_run: o[i] =
  // cast(a[i]) on ATen's thread pool. `Cast` is a per-arm closure type
  // rather than a function pointer so the cast inlines into the loop body
  // (a function pointer measured 18% slower), `IsSigned` is a template
  // parameter so the unsigned early return folds away on the signed path
  // (2.5x on the loop), and `p` is a SuperfpParams built once per tensor
  // rather than re-derived per element. The cast runs in the carrier,
  // carrier_t<scalar_t>: binary64 for float64, binary32 otherwise.
  template <typename scalar_t, bool IsSigned, class Cast>
  void superfp_run(const scalar_t *a, scalar_t *o, int64_t size,
                   const SuperfpParamsT<carrier_t<scalar_t>> &p, Cast cast)
  {
    quant_kernel(a, o, size,
                 [=](scalar_t x) -> scalar_t
                 {
                   return static_cast<scalar_t>(
                       cast(static_cast<carrier_t<scalar_t>>(x), IsSigned, p));
                 });
  }

  // Deterministic-mode driver: builds the cast's parameters (the region
  // cutoffs between normal, supernormal and underflow, and the rounding
  // masks) once per tensor and dispatches the round mode outside the loop,
  // one closure type per arm.
  template <typename scalar_t, bool IsSigned>
  void superfp_kernel_impl(const scalar_t *a, scalar_t *o, int64_t size, int man_bits,
                           int exp_bits, int normal_binades, int bias,
                           RoundMode round_mode, SaturationMode saturation_mode)
  {
    const SuperfpParamsT<carrier_t<scalar_t>> p = make_superfp_params<carrier_t<scalar_t>>(
        man_bits, exp_bits, normal_binades, bias, saturation_mode);

    switch (round_mode)
    {
    case RoundMode::RNE:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_superfp_nearest_even(v, sg, q); });
      break;

    case RoundMode::RNA:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_superfp_nearest_away(v, sg, q); });
      break;

    case RoundMode::RU:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_superfp_up(v, sg, q); });
      break;

    case RoundMode::RD:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_superfp_down(v, sg, q); });
      break;

    case RoundMode::RZ:
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_superfp_zero(v, sg, q); });
      break;

    default: // RO
      superfp_run<scalar_t, IsSigned>(
          a, o, size, p,
          [](auto v, bool sg, const auto &q) { return cast_superfp_odd(v, sg, q); });
      break;
    }
  }

  // Stochastic-rounding driver: the same parameter build, with
  // quant_kernel_sr (utils.h) handing each element a random word of the
  // carrier's width, keyed on the element's index so the result does not
  // depend on the thread count. The cast takes prng_bits of that word as
  // the rounding offset.
  template <typename scalar_t, bool IsSigned>
  void superfp_kernel_sr_impl(const scalar_t *a, scalar_t *o, int64_t size,
                              int man_bits, int exp_bits, int normal_binades, int bias,
                              int prng_bits, uint64_t seed, SaturationMode saturation_mode)
  {
    const SuperfpParamsT<carrier_t<scalar_t>> p = make_superfp_params<carrier_t<scalar_t>>(
        man_bits, exp_bits, normal_binades, bias, saturation_mode);

    quant_kernel_sr(a, o, size, seed,
                    [=](scalar_t x, typename FloatTraits<carrier_t<scalar_t>>::word_t rv) -> scalar_t
                    {
                      return static_cast<scalar_t>(cast_superfp_stochastic(
                          static_cast<carrier_t<scalar_t>>(x), rv, prng_bits, IsSigned, p));
                    });
  }

  // The body of both entry points, as binaryK_quantize_into is (binaryK_kernel.cpp):
  // `o` is `a` itself for the in-place op. Dispatches once on the storage
  // dtype and once on the sign, and draws one seed only when the round mode
  // is SR.
  void superfp_quantize_into(const Tensor &a, Tensor &o, int64_t man_bits, int64_t exp_bits,
                             int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed,
                             int64_t round_mode, int64_t saturation_mode)
  {
    const int64_t size = a.numel(); // int would truncate past 2^31 elements
    RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
    SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

    const int man_bits_ = static_cast<int>(man_bits);
    const int exp_bits_ = static_cast<int>(exp_bits);
    const int normal_binades_ = static_cast<int>(normal_binades);
    const int bias_ = static_cast<int>(bias);
    const int prng_bits_ = static_cast<int>(prng_bits);

    if (round_mode_ != RoundMode::SR)
    {
      MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_quantize_cpu", [&]
                                      {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();
        if (is_signed)
          superfp_kernel_impl<scalar_t, true>(p_a, p_o, size, man_bits_, exp_bits_, normal_binades_,
                         bias_, round_mode_, saturation_mode_);
        else
          superfp_kernel_impl<scalar_t, false>(p_a, p_o, size, man_bits_, exp_bits_, normal_binades_,
                         bias_, round_mode_, saturation_mode_); });
    }
    else
    {
      const uint64_t seed = draw_cpu_seed();
      MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_quantize_cpu_sr", [&]
                                      {
        const scalar_t *p_a = a.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();
        if (is_signed)
          superfp_kernel_sr_impl<scalar_t, true>(p_a, p_o, size, man_bits_, exp_bits_,
                         normal_binades_, bias_, prng_bits_, seed, saturation_mode_);
        else
          superfp_kernel_sr_impl<scalar_t, false>(p_a, p_o, size, man_bits_, exp_bits_,
                         normal_binades_, bias_, prng_bits_, seed, saturation_mode_); });
    }
  }

} // namespace

// The CPU kernel behind mptorch::superfp_quant: a new tensor of a's shape and
// dtype, every element rounded to the superfp format (man_bits, exp_bits,
// normal_binades, bias, is_signed) in the given round and saturation modes
// (the int64_t arguments are the enum values of common/modes.h). superfp
// takes no subnormals mode: its supernormal binades occupy the encoding
// space subnormals would.
Tensor superfp_quantize_cpu(Tensor a, int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                            int64_t bias, int64_t prng_bits, bool is_signed,
                            int64_t round_mode, int64_t saturation_mode)
{
  // data_ptr() walks storage linearly, so a strided input is copied to a
  // contiguous one first (a no-op for an input that already is).
  auto a_c = a.contiguous();
  auto o = empty_like(a_c);
  superfp_quantize_into(a_c, o, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                        round_mode, saturation_mode);
  return o;
}

// The CPU kernel behind mptorch::superfp_quant_: the same rounding written
// over `a`, which must be contiguous, as in binaryK_quantize_cpu_.
Tensor &superfp_quantize_cpu_(Tensor &a, int64_t man_bits, int64_t exp_bits,
                              int64_t normal_binades, int64_t bias, int64_t prng_bits,
                              bool is_signed, int64_t round_mode, int64_t saturation_mode)
{
  TORCH_CHECK(a.is_contiguous(), "superfp_quant_ writes its argument in place and needs a "
              "contiguous tensor, got strides ", a.strides(), " for sizes ", a.sizes(),
              ": use superfp_quant, which copies a strided input");
  superfp_quantize_into(a, a, man_bits, exp_bits, normal_binades, bias, prng_bits, is_signed,
                        round_mode, saturation_mode);
  return a;
}
