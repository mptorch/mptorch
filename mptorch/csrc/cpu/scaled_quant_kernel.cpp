// The CPU kernels of the three scaled quantizers: binaryK_quant_scaled,
// superfp_quant_scaled and fixedpoint_quant_scaled. Each element is
// multiplied by its scale, rounded to the format and divided by the scale
// again, in one pass (common/scaled_quant_host.h states the operation and the
// layout of the scales). They are a file of their own, with drivers of their
// own, so that the elementwise quantizers' objects are what they were.
//
// The multiply and the divide are plain carrier operations, rounded to
// nearest: setup.py's -ffp-contract=off keeps the compiler from fusing the
// multiply into the cast's own arithmetic, so each rounds as torch's
// elementwise mul and div do in mptorch.quant.ScaledQuant's unfused
// spelling, which these kernels reproduce bit for bit.

#include "../common/cast_binaryK.h"
#include "../common/cast_fixedpoint.h"
#include "../common/cast_superfp.h"
#include "../common/dispatch.h"
#include "../common/modes.h"
#include "../common/scaled_quant_host.h"
#include "../common/stochastic_host.h"
#include "utils.h"
#include <ATen/ops/empty_like.h>
#include <algorithm>
#include <type_traits>

using namespace at;

namespace
{

  // o[i] = quant(a[i], s) for i in [0, size), where s is element i's scale
  // (scale_layout, common/scaled_quant_host.h), on ATen's thread pool. A chunk
  // walks its range one run of a shared scale at a time, so the inner loop is
  // quant_kernel's (cpu/utils.h) with the scale a constant of the run.
  template <typename scalar_t, class Quant>
  void scaled_kernel(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                     mptorch::ScaleLayout l, Quant quant)
  {
    at::parallel_for(0, size, mptorch_cpu::quant_grain_size, [=](int64_t begin, int64_t end)
                     {
      int64_t row = begin / l.cols;
      int64_t col = begin - row * l.cols;
      for (int64_t i = begin; i < end;)
      {
        const int64_t blk = col / l.block;
        const int64_t stop = i + std::min(end - i, std::min((blk + 1) * l.block, l.cols) - col);
        const carrier_t<scalar_t> sv = s[row * l.per_row + blk];
        for (int64_t k = i; k < stop; ++k)
          o[k] = quant(a[k], sv);
        col += stop - i;
        i = stop;
        if (col == l.cols)
        {
          col = 0;
          ++row;
        }
      } });
  }

  // The RoundMode::SR twin: element i also takes the random word that
  // quant_kernel_sr (cpu/utils.h) gives element i of a call drawing `seed`,
  // word i & 3 of Philox block i >> 2 (a float64 element the 64-bit word
  // 2 * (i & 1) of block i >> 1), so the result is that of the elementwise
  // quantizer on the scaled tensor, whatever the thread count.
  template <typename scalar_t, class Quant>
  void scaled_kernel_sr(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                        mptorch::ScaleLayout l, uint64_t seed, Quant quant)
  {
    at::parallel_for(0, size, mptorch_cpu::quant_grain_size, [=](int64_t begin, int64_t end)
                     {
      PhiloxBlock blk_words;
      int64_t row = begin / l.cols;
      int64_t col = begin - row * l.cols;
      for (int64_t i = begin; i < end;)
      {
        const int64_t blk = col / l.block;
        const int64_t stop = i + std::min(end - i, std::min((blk + 1) * l.block, l.cols) - col);
        const carrier_t<scalar_t> sv = s[row * l.per_row + blk];
        for (int64_t k = i; k < stop; ++k)
        {
          if constexpr (std::is_same_v<scalar_t, double>)
          {
            if (k == begin || (k & 1) == 0)
              blk_words = philox_block(seed, (uint64_t)(k >> 1), 0);
            o[k] = quant(a[k], sv, blk_words.word64(2 * (int)(k & 1)));
          }
          else
          {
            if (k == begin || (k & 3) == 0)
              blk_words = philox_block(seed, (uint64_t)(k >> 2), 0);
            o[k] = quant(a[k], sv, blk_words.word((int)(k & 3)));
          }
        }
        col += stop - i;
        i = stop;
        if (col == l.cols)
        {
          col = 0;
          ++row;
        }
      } });
  }

  template <typename scalar_t>
  using word_of = typename FloatTraits<carrier_t<scalar_t>>::word_t;

  // --- binaryK ---------------------------------------------------------------

  template <typename scalar_t, bool IsSigned, class Cast>
  void binaryK_scaled_run(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                          mptorch::ScaleLayout l, SubnormalsMode sm,
                          const BinaryKParamsT<carrier_t<scalar_t>> &p, Cast cast)
  {
    using T = carrier_t<scalar_t>;
    scaled_kernel(a, o, size, s, l, [=](scalar_t x, T sv) -> scalar_t
                  { return static_cast<scalar_t>(cast(static_cast<T>(x) * sv, IsSigned, sm, p) / sv); });
  }

  template <typename scalar_t, bool IsSigned>
  void binaryK_scaled_impl(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                           mptorch::ScaleLayout l, int K, int P, int bias, RoundMode round_mode,
                           SaturationMode saturation_mode, SubnormalsMode sm)
  {
    const BinaryKParamsT<carrier_t<scalar_t>> p = make_binaryK_params<carrier_t<scalar_t>>(
        P - 1, IsSigned ? K - P : K - P + 1, bias, IsSigned, saturation_mode,
        sm == SubnormalsMode::EXTENDED_NORMALS);
    switch (round_mode)
    {
    case RoundMode::RNE:
      binaryK_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, sm, p, [](auto v, bool sg, SubnormalsMode m, const auto &q)
                                             { return cast_binaryK_nearest_even(v, sg, m, q); });
      break;
    case RoundMode::RNA:
      binaryK_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, sm, p, [](auto v, bool sg, SubnormalsMode m, const auto &q)
                                             { return cast_binaryK_nearest_away(v, sg, m, q); });
      break;
    case RoundMode::RU:
      binaryK_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, sm, p, [](auto v, bool sg, SubnormalsMode m, const auto &q)
                                             { return cast_binaryK_up(v, sg, m, q); });
      break;
    case RoundMode::RD:
      binaryK_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, sm, p, [](auto v, bool sg, SubnormalsMode m, const auto &q)
                                             { return cast_binaryK_down(v, sg, m, q); });
      break;
    case RoundMode::RZ:
      binaryK_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, sm, p, [](auto v, bool sg, SubnormalsMode m, const auto &q)
                                             { return cast_binaryK_zero(v, sg, m, q); });
      break;
    default: // RO
      binaryK_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, sm, p, [](auto v, bool sg, SubnormalsMode m, const auto &q)
                                             { return cast_binaryK_odd(v, sg, m, q); });
      break;
    }
  }

  template <typename scalar_t, bool IsSigned>
  void binaryK_scaled_sr_impl(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                              mptorch::ScaleLayout l, int K, int P, int bias, int prng_bits, uint64_t seed,
                              SaturationMode saturation_mode, SubnormalsMode sm)
  {
    using T = carrier_t<scalar_t>;
    const BinaryKParamsT<T> p = make_binaryK_params<T>(
        P - 1, IsSigned ? K - P : K - P + 1, bias, IsSigned, saturation_mode,
        sm == SubnormalsMode::EXTENDED_NORMALS);
    const StochasticT<T> st = make_stochastic_binaryK<T>(p, sm, IsSigned, prng_bits);
    scaled_kernel_sr(a, o, size, s, l, seed, [=](scalar_t x, T sv, word_of<scalar_t> rv) -> scalar_t
                     { return static_cast<scalar_t>(
                           cast_binaryK_stochastic(static_cast<T>(x) * sv, rv, st, IsSigned, sm, p) / sv); });
  }

  // --- superfp ---------------------------------------------------------------

  template <typename scalar_t, bool IsSigned, class Cast>
  void superfp_scaled_run(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                          mptorch::ScaleLayout l, const SuperfpParamsT<carrier_t<scalar_t>> &p, Cast cast)
  {
    using T = carrier_t<scalar_t>;
    scaled_kernel(a, o, size, s, l, [=](scalar_t x, T sv) -> scalar_t
                  { return static_cast<scalar_t>(cast(static_cast<T>(x) * sv, IsSigned, p) / sv); });
  }

  template <typename scalar_t, bool IsSigned>
  void superfp_scaled_impl(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                           mptorch::ScaleLayout l, int man_bits, int exp_bits, int normal_binades, int bias,
                           RoundMode round_mode, SaturationMode saturation_mode)
  {
    const SuperfpParamsT<carrier_t<scalar_t>> p = make_superfp_params<carrier_t<scalar_t>>(
        man_bits, exp_bits, normal_binades, bias, saturation_mode);
    switch (round_mode)
    {
    case RoundMode::RNE:
      superfp_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                             { return cast_superfp_nearest_even(v, sg, q); });
      break;
    case RoundMode::RNA:
      superfp_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                             { return cast_superfp_nearest_away(v, sg, q); });
      break;
    case RoundMode::RU:
      superfp_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                             { return cast_superfp_up(v, sg, q); });
      break;
    case RoundMode::RD:
      superfp_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                             { return cast_superfp_down(v, sg, q); });
      break;
    case RoundMode::RZ:
      superfp_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                             { return cast_superfp_zero(v, sg, q); });
      break;
    default: // RO
      superfp_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                             { return cast_superfp_odd(v, sg, q); });
      break;
    }
  }

  template <typename scalar_t, bool IsSigned>
  void superfp_scaled_sr_impl(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                              mptorch::ScaleLayout l, int man_bits, int exp_bits, int normal_binades,
                              int bias, int prng_bits, uint64_t seed, SaturationMode saturation_mode)
  {
    using T = carrier_t<scalar_t>;
    const SuperfpParamsT<T> p = make_superfp_params<T>(man_bits, exp_bits, normal_binades, bias,
                                                       saturation_mode);
    const StochasticT<T> st = make_stochastic_superfp<T>(p, IsSigned, prng_bits);
    scaled_kernel_sr(a, o, size, s, l, seed, [=](scalar_t x, T sv, word_of<scalar_t> rv) -> scalar_t
                     { return static_cast<scalar_t>(
                           cast_superfp_stochastic(static_cast<T>(x) * sv, rv, st, IsSigned, p) / sv); });
  }

  // --- fixed point -----------------------------------------------------------

  template <typename scalar_t, bool IsSigned, class Cast>
  void fixedpoint_scaled_run(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                             mptorch::ScaleLayout l, const FixedPointParamsT<carrier_t<scalar_t>> &p,
                             Cast cast)
  {
    using T = carrier_t<scalar_t>;
    scaled_kernel(a, o, size, s, l, [=](scalar_t x, T sv) -> scalar_t
                  { return static_cast<scalar_t>(cast(static_cast<T>(x) * sv, IsSigned, p) / sv); });
  }

  template <typename scalar_t, bool IsSigned>
  void fixedpoint_scaled_impl(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                              mptorch::ScaleLayout l, int wl, int fl, bool symmetric, RoundMode round_mode)
  {
    const FixedPointParamsT<carrier_t<scalar_t>> p =
        make_fixedpoint_params<carrier_t<scalar_t>>(wl, fl, IsSigned, symmetric);
    switch (round_mode)
    {
    case RoundMode::RNE:
      fixedpoint_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                                { return cast_fixedpoint_nearest_even(v, sg, q); });
      break;
    case RoundMode::RNA:
      fixedpoint_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                                { return cast_fixedpoint_nearest_away(v, sg, q); });
      break;
    case RoundMode::RU:
      fixedpoint_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                                { return cast_fixedpoint_up(v, sg, q); });
      break;
    case RoundMode::RD:
      fixedpoint_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                                { return cast_fixedpoint_down(v, sg, q); });
      break;
    case RoundMode::RZ:
      fixedpoint_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                                { return cast_fixedpoint_zero(v, sg, q); });
      break;
    default: // RO
      fixedpoint_scaled_run<scalar_t, IsSigned>(a, o, size, s, l, p, [](auto v, bool sg, const auto &q)
                                                { return cast_fixedpoint_odd(v, sg, q); });
      break;
    }
  }

  template <typename scalar_t, bool IsSigned>
  void fixedpoint_scaled_sr_impl(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                                 mptorch::ScaleLayout l, int wl, int fl, bool symmetric, int prng_bits,
                                 uint64_t seed)
  {
    using T = carrier_t<scalar_t>;
    const FixedPointParamsT<T> p = make_fixedpoint_params<T>(wl, fl, IsSigned, symmetric);
    scaled_kernel_sr(a, o, size, s, l, seed, [=](scalar_t x, T sv, word_of<scalar_t> rv) -> scalar_t
                     { return static_cast<scalar_t>(
                           cast_fixedpoint_stochastic(static_cast<T>(x) * sv, rv, prng_bits, IsSigned, p) / sv); });
  }

  // SR_B and SR_C run as SR, with the variant recorded in prng_bits
  // (common/stochastic_host.h), as in every quantizer.
  int64_t canonical_round_mode(int64_t round_mode, int64_t &prng_bits, const char *op)
  {
    int variant;
    round_mode = mptorch::sr_canonical(round_mode, variant);
    if (variant >= 0)
      prng_bits = mptorch::sr_encode(prng_bits, variant, op);
    return round_mode;
  }

} // namespace

// Each entry point makes its elementwise twin's format checks first, then the
// scale operands' (common/scaled_quant_host.h), and copies a strided input
// as its twin does. An empty tensor returns before any RNG state is drawn.

Tensor binaryK_quantize_scaled_cpu(Tensor a, Tensor scale, int64_t block_size, int64_t K, int64_t P,
                                   int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                                   int64_t saturation_mode, int64_t subnormals_mode)
{
  const char *op = "binaryK_quant_scaled";
  mptorch::check_quantizer_format(
      mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode),
      mptorch::carrier_bits(a.scalar_type() == at::kDouble), prng_bits, round_mode, op);
  mptorch::check_scaled_operands(a, scale, block_size, op);
  auto a_c = a.contiguous();
  auto s_c = scale.contiguous();
  auto o = empty_like(a_c);
  const int64_t size = a_c.numel();
  if (size == 0)
    return o;
  const mptorch::ScaleLayout l = mptorch::scale_layout(a_c, block_size);
  const RoundMode rm = static_cast<RoundMode>(canonical_round_mode(round_mode, prng_bits, op));
  const SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  const SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
  const int K_ = static_cast<int>(K), P_ = static_cast<int>(P), bias_ = static_cast<int>(bias);
  if (rm != RoundMode::SR)
  {
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "binaryK_quant_scaled", [&]
                                 {
      const scalar_t *pa = a_c.data_ptr<scalar_t>();
      scalar_t *po = o.data_ptr<scalar_t>();
      const carrier_t<scalar_t> *ps = s_c.data_ptr<carrier_t<scalar_t>>();
      if (is_signed)
        binaryK_scaled_impl<scalar_t, true>(pa, po, size, ps, l, K_, P_, bias_, rm, sat, sub);
      else
        binaryK_scaled_impl<scalar_t, false>(pa, po, size, ps, l, K_, P_, bias_, rm, sat, sub); });
  }
  else
  {
    const uint64_t seed = draw_cpu_seed();
    const int prng = static_cast<int>(prng_bits);
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "binaryK_quant_scaled", [&]
                                 {
      const scalar_t *pa = a_c.data_ptr<scalar_t>();
      scalar_t *po = o.data_ptr<scalar_t>();
      const carrier_t<scalar_t> *ps = s_c.data_ptr<carrier_t<scalar_t>>();
      if (is_signed)
        binaryK_scaled_sr_impl<scalar_t, true>(pa, po, size, ps, l, K_, P_, bias_, prng, seed, sat, sub);
      else
        binaryK_scaled_sr_impl<scalar_t, false>(pa, po, size, ps, l, K_, P_, bias_, prng, seed, sat, sub); });
  }
  return o;
}

Tensor superfp_quantize_scaled_cpu(Tensor a, Tensor scale, int64_t block_size, int64_t man_bits,
                                   int64_t exp_bits, int64_t normal_binades, int64_t bias,
                                   int64_t prng_bits, bool is_signed, int64_t round_mode,
                                   int64_t saturation_mode)
{
  const char *op = "superfp_quant_scaled";
  mptorch::check_quantizer_format(
      mptorch::superfp_format(man_bits, exp_bits, normal_binades, bias, saturation_mode),
      mptorch::carrier_bits(a.scalar_type() == at::kDouble), prng_bits, round_mode, op);
  mptorch::check_scaled_operands(a, scale, block_size, op);
  auto a_c = a.contiguous();
  auto s_c = scale.contiguous();
  auto o = empty_like(a_c);
  const int64_t size = a_c.numel();
  if (size == 0)
    return o;
  const mptorch::ScaleLayout l = mptorch::scale_layout(a_c, block_size);
  const RoundMode rm = static_cast<RoundMode>(canonical_round_mode(round_mode, prng_bits, op));
  const SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  const int m = static_cast<int>(man_bits), e = static_cast<int>(exp_bits);
  const int nb = static_cast<int>(normal_binades), b = static_cast<int>(bias);
  if (rm != RoundMode::SR)
  {
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "superfp_quant_scaled", [&]
                                 {
      const scalar_t *pa = a_c.data_ptr<scalar_t>();
      scalar_t *po = o.data_ptr<scalar_t>();
      const carrier_t<scalar_t> *ps = s_c.data_ptr<carrier_t<scalar_t>>();
      if (is_signed)
        superfp_scaled_impl<scalar_t, true>(pa, po, size, ps, l, m, e, nb, b, rm, sat);
      else
        superfp_scaled_impl<scalar_t, false>(pa, po, size, ps, l, m, e, nb, b, rm, sat); });
  }
  else
  {
    const uint64_t seed = draw_cpu_seed();
    const int prng = static_cast<int>(prng_bits);
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "superfp_quant_scaled", [&]
                                 {
      const scalar_t *pa = a_c.data_ptr<scalar_t>();
      scalar_t *po = o.data_ptr<scalar_t>();
      const carrier_t<scalar_t> *ps = s_c.data_ptr<carrier_t<scalar_t>>();
      if (is_signed)
        superfp_scaled_sr_impl<scalar_t, true>(pa, po, size, ps, l, m, e, nb, b, prng, seed, sat);
      else
        superfp_scaled_sr_impl<scalar_t, false>(pa, po, size, ps, l, m, e, nb, b, prng, seed, sat); });
  }
  return o;
}

Tensor fixedpoint_quantize_scaled_cpu(Tensor a, Tensor scale, int64_t block_size, int64_t wl,
                                      int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric,
                                      int64_t round_mode)
{
  const char *op = "fixedpoint_quant_scaled";
  mptorch::check_quantizer_format(mptorch::fixedpoint_format(wl - (is_signed ? 1 : 0), fl),
                                  mptorch::carrier_bits(a.scalar_type() == at::kDouble), prng_bits,
                                  round_mode, op);
  mptorch::check_scaled_operands(a, scale, block_size, op);
  auto a_c = a.contiguous();
  auto s_c = scale.contiguous();
  auto o = empty_like(a_c);
  const int64_t size = a_c.numel();
  if (size == 0)
    return o;
  const mptorch::ScaleLayout l = mptorch::scale_layout(a_c, block_size);
  const RoundMode rm = static_cast<RoundMode>(canonical_round_mode(round_mode, prng_bits, op));
  const int wl_ = static_cast<int>(wl), fl_ = static_cast<int>(fl);
  if (rm != RoundMode::SR)
  {
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "fixedpoint_quant_scaled", [&]
                                 {
      const scalar_t *pa = a_c.data_ptr<scalar_t>();
      scalar_t *po = o.data_ptr<scalar_t>();
      const carrier_t<scalar_t> *ps = s_c.data_ptr<carrier_t<scalar_t>>();
      if (is_signed)
        fixedpoint_scaled_impl<scalar_t, true>(pa, po, size, ps, l, wl_, fl_, symmetric, rm);
      else
        fixedpoint_scaled_impl<scalar_t, false>(pa, po, size, ps, l, wl_, fl_, symmetric, rm); });
  }
  else
  {
    const uint64_t seed = draw_cpu_seed();
    const int prng = static_cast<int>(prng_bits);
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "fixedpoint_quant_scaled", [&]
                                 {
      const scalar_t *pa = a_c.data_ptr<scalar_t>();
      scalar_t *po = o.data_ptr<scalar_t>();
      const carrier_t<scalar_t> *ps = s_c.data_ptr<carrier_t<scalar_t>>();
      if (is_signed)
        fixedpoint_scaled_sr_impl<scalar_t, true>(pa, po, size, ps, l, wl_, fl_, symmetric, prng, seed);
      else
        fixedpoint_scaled_sr_impl<scalar_t, false>(pa, po, size, ps, l, wl_, fl_, symmetric, prng, seed); });
  }
  return o;
}
