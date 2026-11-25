#pragma once

// The host side of the fixed-point GEMM ops (common/fixedpoint_gemm.h): the
// schema's formats, the checks a call must pass, the rule that picks the
// carrier path or the integer accumulator, and the driver.
//
// The driver is run_custom_matmul's steps (gemm_host.h) in run_custom_matmul's
// order, with two differences, which is why it is a function of its own
// rather than a branch in that one (an edit there would move the host code
// of every GEMM op): the Args are one of five types, chosen per call; and
// the result is float64 when the accumulate format has more significant
// bits than the operands' dtype holds (the wide result, which the layers
// round once with their output rounding before storing it in that dtype).
//
// mptorch/number/fixedpoint.py's _fixedpoint_gemm and _fixedpoint_gemm_wide
// are this rule's Python twin, which the Python layer applies first, with
// messages that name the formats; the checks here are what a direct
// torch.ops call meets, and what keeps every shift and conversion in the
// kernels defined.

#include "fixedpoint_gemm.h"
#include "gemm_args.h"
#include "gemm_host.h"
#include "stochastic_host.h"
#include <ATen/ops/empty.h>
#include <cstdint>

namespace mptorch::gemm
{
  // A fixed-point format as the schema gives it: its magnitude bits (wl less
  // the sign bit, computed in 128 bits), fl and prng_bits, each narrowed to
  // int with saturation (gemm_args.h's narrow_to_int), so that a value past
  // int's range reaches the checks as INT_MIN or INT_MAX, which each of them
  // refuses and prints as "or less" or "or more" (format_host.h's Shown).
  struct FixedPointFormat
  {
    int mag_bits = 0, fl = 0;
    bool is_signed = true, symmetric = false;
    int prng_bits = 0;

    int64_t mag() const { return mag_bits; }
    // the word length, for a format the checks have passed
    int wl() const { return mag_bits + (is_signed ? 1 : 0); }
  };

  inline FixedPointFormat pack_fixedpoint(int64_t wl, int64_t fl, bool is_signed, bool symmetric,
                                          int64_t prng_bits)
  {
    return FixedPointFormat{narrow_to_int(static_cast<__int128>(wl) - (is_signed ? 1 : 0)),
                            narrow_to_int(fl), is_signed, symmetric, narrow_to_int(prng_bits)};
  }

  // One call: a split mac (mul, then acc) or a fused one (acc alone), the
  // accumulate format's overflow, and the test hook that runs the integer
  // accumulator where the carrier path would give the same bits.
  struct FixedPointGemmCall
  {
    bool fused = false;
    FixedPointFormat mul{}, acc{}; // mul unused when fused
    bool wrap = false;
    bool integer = false;
  };

  inline FixedPointGemmCall pack_fixedpoint_split(int64_t mul_wl, int64_t mul_fl,
                                                  bool mul_is_signed, bool mul_symmetric,
                                                  int64_t acc_wl, int64_t acc_fl,
                                                  bool acc_is_signed, bool acc_symmetric,
                                                  int64_t mul_prng_bits, int64_t acc_prng_bits,
                                                  bool wrap, bool integer)
  {
    FixedPointGemmCall c;
    c.mul = pack_fixedpoint(mul_wl, mul_fl, mul_is_signed, mul_symmetric, mul_prng_bits);
    c.acc = pack_fixedpoint(acc_wl, acc_fl, acc_is_signed, acc_symmetric, acc_prng_bits);
    c.wrap = wrap;
    c.integer = integer;
    return c;
  }

  inline FixedPointGemmCall pack_fixedpoint_fused(int64_t fma_wl, int64_t fma_fl,
                                                  bool fma_is_signed, bool fma_symmetric,
                                                  int64_t fma_prng_bits, bool wrap)
  {
    FixedPointGemmCall c;
    c.fused = true;
    c.acc = pack_fixedpoint(fma_wl, fma_fl, fma_is_signed, fma_symmetric, fma_prng_bits);
    c.wrap = wrap;
    return c;
  }

  // The carrier's bounds the rule reads, as mptorch/number/_carrier.py's
  // _Carrier states them: the significand's precision and stored bits, the
  // exponents of the smallest normal and the largest finite binade, and the
  // lowest step the fixed-point cast rounds correctly.
  struct FixedPointCarrierBounds
  {
    const char *name;
    int64_t precision, man_bits, min_normal_exp, top_exp, min_simulable_exp;
  };

  inline FixedPointCarrierBounds fixedpoint_carrier_bounds(bool binary64)
  {
    return binary64 ? FixedPointCarrierBounds{"binary64", 53, 52, -1022, 1023, -1021}
                    : FixedPointCarrierBounds{"binary32", 24, 23, -126, 127, -125};
  }

  // The integer accumulator's limits (mptorch/number/fixedpoint.py's
  // _INTEGER_ACC_BITS and _INTEGER_SPAN_BITS): an accumulate format of at
  // most 53 magnitude bits, whose values float64 holds, and sums of at most
  // 62 bits in its int64.
  constexpr int64_t FIXEDPOINT_ACC_BITS = 53;
  constexpr int64_t FIXEDPOINT_SPAN_BITS = 62;

  // The significant bits an exact sum of one value of each format can need:
  // a multiple of the finer step, below 2^(max(mag_bits - fl) + 1).
  inline int64_t fixedpoint_span(const FixedPointFormat &a, const FixedPointFormat &b)
  {
    const int64_t top = std::max(a.mag() - a.fl, b.mag() - b.fl) + 1;
    return top + std::max<int64_t>(a.fl, b.fl);
  }

  // The exponent of the leading bit of a format's end farther from zero:
  // the top, or in two's complement the bottom, -2^(mag_bits - fl), a
  // binade higher.
  inline int64_t fixedpoint_end_exp(const FixedPointFormat &f)
  {
    return f.mag() - 1 - f.fl + (f.is_signed && !f.symmetric ? 1 : 0);
  }

  // Whether the carrier's cast rounds every value of the format correctly,
  // with `prng_bits` random bits when `stochastic`: what the elementwise
  // quantizer's carrier rule (number/fixedpoint.py's _fixedpoint_findings)
  // finds neither an error nor a warning in.
  inline bool fixedpoint_in_carrier(const FixedPointFormat &f, const FixedPointCarrierBounds &c,
                                    bool stochastic)
  {
    return f.mag() <= c.precision && (!stochastic || f.mag() + f.prng_bits <= c.precision) &&
           f.mag() - 1 - f.fl >= c.min_normal_exp && fixedpoint_end_exp(f) <= c.top_exp &&
           -static_cast<int64_t>(f.fl) >= c.min_simulable_exp;
  }

  // How a checked call runs: on the carrier path (FixedPointSplitArgs) or
  // the integer accumulator, and whether its result is float64 for operands
  // of a narrower dtype.
  struct FixedPointGemmPlan
  {
    bool carrier_path = false;
    bool wide = false;
  };

  // The significand precision of a GEMM operand dtype, or 0 for a dtype no
  // GEMM takes (gemm_dtype_of refuses it after the allocation, as in
  // run_custom_matmul).
  inline int64_t fixedpoint_storage_precision(at::ScalarType t)
  {
    switch (t)
    {
    case at::kDouble:
      return 53;
    case at::kFloat:
      return 24;
    case at::kHalf:
      return 11;
    case at::kBFloat16:
      return 8;
    default:
      return 0;
    }
  }

  // Checks a call, in the order of the Python twin's refusals, then
  // canonicalizes round_mode in place (SR_B and SR_C run as SR, the variant
  // in each format's prng_bits, stochastic_host.h's sr_canonical), encodes
  // the formats' random bits, and returns the plan. `dtype` is the operands'.
  inline FixedPointGemmPlan plan_fixedpoint_gemm(FixedPointGemmCall &call, int64_t &round_mode,
                                                 at::ScalarType dtype, const char *op)
  {
    const FixedPointCarrierBounds c = fixedpoint_carrier_bounds(dtype == at::kDouble);
    const CarrierBits cb = carrier_bits(dtype == at::kDouble);
    const bool stochastic = is_stochastic_mode(round_mode);
    FixedPointFormat &mul = call.mul, &acc = call.acc;

    // The multiply format is rounded by the carrier's cast on both paths, so
    // it must be one the carrier rounds correctly, every value and step: an
    // end past the carrier would make a rounded product infinite, and a step
    // below the cast's floor misrounds near zero, where the definition is
    // exact.
    if (!call.fused)
    {
      check_format(Format{FormatFamily::FixedPoint, mul.mag(), 0, 0, 0, mul.fl, 0, 0, true}, cb,
                   op);
      TORCH_CHECK(
          fixedpoint_end_exp(mul) <= c.top_exp, op,
          ": the multiply format's end farther from zero is ", "in the 2^", fixedpoint_end_exp(mul),
          " binade, past ", c.name, "'s range, which ends below 2^", c.top_exp + 1,
          ": the multiply format is rounded in the carrier, which must represent both of its ends");
      TORCH_CHECK(-static_cast<int64_t>(mul.fl) >= c.min_simulable_exp, op,
                  ": the multiply format's step is 2^", -static_cast<int64_t>(mul.fl), ", below 2^",
                  c.min_simulable_exp, ", the lowest ", c.name,
                  "'s cast rounds correctly: the multiply format is rounded in the carrier");
      if (stochastic)
        sr_check_prng(mul.prng_bits, mul.mag(), true, cb, op, true);
    }

    // The accumulate format is the integer accumulator's: at most 53
    // magnitude bits, and values float64 holds.
    TORCH_CHECK(acc.mag() >= 1 && acc.mag() <= FIXEDPOINT_ACC_BITS, op,
                ": the accumulate format has ", Shown{acc.mag(), true},
                " magnitude bits, and a fixed-point GEMM accumulates 1 to ", FIXEDPOINT_ACC_BITS,
                ", which float64 holds exactly");
    TORCH_CHECK(acc.fl >= acc.mag() - EXP_LIMIT && acc.fl <= EXP_LIMIT, op,
                ": the accumulate format's fl is ", Shown{acc.fl, true}, ", but with ", acc.mag(),
                " magnitude bits the kernels take ", acc.mag() - EXP_LIMIT, " to ", EXP_LIMIT);
    TORCH_CHECK(
        acc.mag() - 1 - acc.fl >= -1022 && fixedpoint_end_exp(acc) <= 1023, op,
        ": the accumulate format's values are not float64 values (its leading bits are in the 2^",
        acc.mag() - 1 - acc.fl, " to 2^", fixedpoint_end_exp(acc),
        " binades): the result is held in float64");
    if (stochastic)
      TORCH_CHECK(acc.prng_bits >= 0 && acc.prng_bits <= c.man_bits, op,
                  ": the accumulate format draws ", Shown{acc.prng_bits, true},
                  " random bits, and a ", c.name, " draw holds ", c.man_bits,
                  " below its sign and exponent");
    TORCH_CHECK(!(call.wrap && acc.is_signed && acc.symmetric), op,
                ": the accumulate format is symmetric, so it has no modulus to wrap around: drop "
                "symmetric, or saturate");
    if (!call.fused)
    {
      const int64_t span = fixedpoint_span(mul, acc);
      TORCH_CHECK(span <= FIXEDPOINT_SPAN_BITS, op,
                  ": a sum of the multiply and accumulate formats needs ", span,
                  " bits, and the integer accumulator holds ", FIXEDPOINT_SPAN_BITS,
                  ": narrow the formats, or bring their steps closer");
    }
    else if (stochastic)
    {
      // the carrier's product, aligned with N + 1 guard bits beside the sum
      const int64_t span = acc.mag() + acc.prng_bits + 4;
      TORCH_CHECK(span <= FIXEDPOINT_SPAN_BITS, op,
                  ": a fused step, with its guard and random bits, needs ", span,
                  " bits, and the integer accumulator holds ", FIXEDPOINT_SPAN_BITS);
    }

    // The plan reads the random bits as N, so it is made before they are
    // encoded with the variant.
    FixedPointGemmPlan plan;
    const int64_t storage = fixedpoint_storage_precision(dtype);
    plan.wide = storage != 0 && acc.mag() > storage;
    // the carrier path stores in the operands' dtype, so a wide result runs
    // the integer accumulator, whose kernels store float64
    plan.carrier_path = !call.fused && !call.wrap && !call.integer && !plan.wide &&
                        fixedpoint_in_carrier(acc, c, stochastic) &&
                        fixedpoint_span(mul, acc) <= c.precision;

    int variant;
    round_mode = sr_canonical(round_mode, variant);
    if (variant >= 0)
    {
      if (!call.fused)
        mul.prng_bits = sr_encode_narrowed(mul.prng_bits, variant, op);
      acc.prng_bits = sr_encode_narrowed(acc.prng_bits, variant, op);
    }
    return plan;
  }

  inline FixedPointWidths fixedpoint_widths(const FixedPointFormat &f)
  {
    return FixedPointWidths{f.wl(), f.fl};
  }

  inline FixedPointCommon fixedpoint_common(const FixedPointFormat &f)
  {
    return FixedPointCommon{f.is_signed, f.symmetric, f.prng_bits};
  }

  template <class Args>
  Args fixedpoint_args(const FixedPointGemmCall &call, const FixedPointGemmPlan &plan)
  {
    Args args;
    args.mul = fixedpoint_widths(call.mul);
    args.acc = fixedpoint_widths(call.acc);
    args.mul_c = fixedpoint_common(call.mul);
    args.acc_c = fixedpoint_common(call.acc);
    if constexpr (requires { args.wide; })
      args.wide = plan.wide;
    return args;
  }

  template <class Backend, class Args>
  void launch_fixedpoint_as(GemmShape &s, const Args &args, const Tensor &a, const Tensor &b,
                            const Tensor &c)
  {
    typename Backend::LaunchContext ctx = Backend::make_context(
        s.use_rng, draws_per_k_step_of(args) * words_per_draw(s.dt) * static_cast<uint64_t>(s.K));
    bind_tensors(ctx, a, b, c, nullptr);
    Backend::launch(s, args, ctx);
  }

  // Launches the plan's Args: the carrier path, or the integer accumulator,
  // split or fused, saturating or wrapping.
  template <class Backend>
  void launch_fixedpoint(GemmShape &s, const FixedPointGemmCall &call,
                         const FixedPointGemmPlan &plan, const Tensor &a, const Tensor &b,
                         const Tensor &c)
  {
    if (call.fused)
    {
      if (call.wrap)
        launch_fixedpoint_as<Backend>(s, fixedpoint_args<FixedPointIntFusedArgs<true>>(call, plan),
                                      a, b, c);
      else
        launch_fixedpoint_as<Backend>(s, fixedpoint_args<FixedPointIntFusedArgs<false>>(call, plan),
                                      a, b, c);
    }
    else if (plan.carrier_path)
      launch_fixedpoint_as<Backend>(s, fixedpoint_args<FixedPointSplitArgs>(call, plan), a, b, c);
    else if (call.wrap)
      launch_fixedpoint_as<Backend>(s, fixedpoint_args<FixedPointIntSplitArgs<true>>(call, plan), a,
                                    b, c);
    else
      launch_fixedpoint_as<Backend>(s, fixedpoint_args<FixedPointIntSplitArgs<false>>(call, plan),
                                    a, b, c);
  }

  // run_custom_matmul (gemm_host.h) for a fixed-point call, step for step,
  // with the result's dtype from the plan.
  template <class Backend>
  Tensor run_custom_matmul_fixedpoint(const char *op_name, const FixedPointGemmCall &call_args,
                                      Tensor a, Tensor b, bool trans_a, bool trans_b,
                                      int64_t round_mode)
  {
    check_matmul_inputs(a, b, op_name, round_mode,
                        static_cast<int64_t>(AccumulateAlgorithm::NAIVE));
    FixedPointGemmCall call = call_args;
    const FixedPointGemmPlan plan =
        plan_fixedpoint_gemm(call, round_mode, a.scalar_type(), op_name);

    int64_t M, K, N, batch, stride_a, stride_b;
    bool batched;
    matmul_output_shape(a, b, trans_a, trans_b, op_name, M, K, N, batch, stride_a, stride_b,
                        batched);

    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    Tensor c = at::empty(matmul_output_sizes(batch, M, N, batched),
                         a.options().dtype(plan.wide ? at::kDouble : a.scalar_type()));
    if (batch == 0 || M == 0 || N == 0)
      return c;

    GemmShape s;
    s.M = M;
    s.K = K;
    s.N = N;
    s.batch = batch;
    s.stride_a = stride_a;
    s.stride_b = stride_b;
    s.trans_a = trans_a;
    s.trans_b = trans_b;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);
    s.dt = mptorch::gemm_dtype_of(a_c, b_c, op_name);
    s.a = a_c.data_ptr();
    s.b = b_c.data_ptr();
    s.c = c.data_ptr();

    launch_fixedpoint<Backend>(s, call, plan, a_c, b_c, c);
    return c;
  }
} // namespace mptorch::gemm
