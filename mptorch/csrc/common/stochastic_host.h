#pragma once

// The host side of the three stochastic modes (common/stochastic.h): where a
// call's RoundMode::SR_B or SR_C becomes RoundMode::SR, with the variant
// recorded next to N in the prng_bits of each of the call's formats. Only the
// entry points and the drivers do this, before the round mode becomes a
// template argument, so dispatch_round_mode and every `== RoundMode::SR`
// test after it see a single stochastic mode, and each kernel has a single
// stochastic instantiation.

#include "format_host.h"
#include "gemm_dtype.h"
#include "stochastic.h"
#include <c10/util/Exception.h>
#include <cstdint>

namespace mptorch
{
  // Whether a validated round mode is stochastic.
  inline bool is_stochastic_mode(int64_t round_mode)
  {
    return round_mode == static_cast<int64_t>(RoundMode::SR) ||
           round_mode == static_cast<int64_t>(RoundMode::SR_B) ||
           round_mode == static_cast<int64_t>(RoundMode::SR_C);
  }

  // Returns the round mode a kernel runs the call in, and sets `variant` to
  // the stochastic variant: 0 for SR, 1 for SR_B, 2 for SR_C, and -1 for a
  // deterministic mode.
  inline int64_t sr_canonical(int64_t round_mode, int &variant)
  {
    variant = is_stochastic_mode(round_mode) ? sr_variant_of(static_cast<RoundMode>(round_mode)) : -1;
    return variant >= 0 ? static_cast<int64_t>(RoundMode::SR) : round_mode;
  }

  // A format's prng_bits under a stochastic call: N, checked to fit in the
  // spec's low byte (a larger value would be read as a variant), with the
  // variant recorded above it. The overload below checks the tighter bound
  // first, so this check is reached only by a field validated elsewhere
  // (checked_block_fmt) or by a format the call does not round with.
  inline int sr_encode(int64_t prng_bits, int variant, const char *op)
  {
    TORCH_CHECK(prng_bits >= 0 && prng_bits <= 255, op, ": prng_bits must be in [0, 255], got ",
                prng_bits);
    return sr_spec(static_cast<int>(prng_bits), variant);
  }

  // Refuses more random bits than fit beside a format in the carrier. The N
  // random bits are drawn directly below the format's last bit, inside the
  // carrier's significand (StochasticT in stochastic.h,
  // cast_fixedpoint_stochastic), so the format's bits and N must fit there
  // together. Past that, the shifts that place the draw go negative, which
  // is undefined behaviour. mptorch.number refuses such formats, so this is
  // what a direct torch.ops call meets. `bits` is the format's stored
  // mantissa bits, or its magnitude bits when `magnitude` is set (a
  // fixed-point format, format_host.h's significand_room). `narrowed` says
  // that a GEMM's packer has narrowed the value to int with saturation, as
  // for format_host.h's Format. Defined out of line (format_checks.cpp), for
  // the reason check_format is.
  void sr_check_prng(int64_t prng_bits, int64_t bits, bool magnitude, CarrierBits c, const char *op,
                     bool narrowed = false);

  // The plain sr_encode above for a GEMM's value, which its packer has
  // narrowed to int with saturation: the refusal prints a value at int's
  // limit as "or more" or "or less" (format_host.h's Shown). It is a
  // function of its own because the elementwise quantizers' bodies inline
  // the plain one, and adding a parameter to it changed their code.
  inline int sr_encode_narrowed(int64_t prng_bits, int variant, const char *op)
  {
    TORCH_CHECK(prng_bits >= 0 && prng_bits <= 255, op, ": prng_bits must be in [0, 255], got ",
                Shown{prng_bits, true});
    return sr_spec(static_cast<int>(prng_bits), variant);
  }

  // A format's prng_bits under a stochastic call, checked against the
  // carrier (sr_check_prng) and encoded. The format itself is checked
  // first, in every mode (check_format).
  inline int sr_encode(int64_t prng_bits, int64_t bits, bool magnitude, CarrierBits c, int variant,
                       const char *op, bool narrowed = false)
  {
    sr_check_prng(prng_bits, bits, magnitude, c, op, narrowed);
    return narrowed ? sr_encode_narrowed(prng_bits, variant, op) : sr_encode(prng_bits, variant, op);
  }

  // The elementwise quantizers' format checks, which their entry points make
  // before anything else (after an in-place op's contiguity check): the
  // format itself in every mode (check_format), and its random bits under a
  // stochastic mode. The body each entry calls next encodes prng_bits with
  // the plain sr_encode above. Defined out of line (format_checks.cpp), for
  // the reason check_format is, and called from the entry points rather than
  // from that body, so that the body, which the compiler inlines its
  // single-thread loops into, is as it was before these checks.
  void check_quantizer_format(const Format &f, CarrierBits c, int64_t prng_bits, int64_t round_mode,
                              const char *op);

  // The widest mantissa among the formats of one role, which bounds the
  // prng_bits they share: the first n_fmt slots of a palette op, or the one
  // format.
  template <class Widths>
  int64_t sr_widest_mantissa(const Widths &widths, int n_fmt)
  {
    int64_t widest = 0;
    for_each_slot(widths, n_fmt, [&](const auto &w) { widest = w.man_bits > widest ? w.man_bits : widest; });
    return widest;
  }

  // Encodes every format of a GEMM's Args (for_each_format_role names the
  // roles). A role's prng_bits is checked against the carrier when the call
  // rounds with its formats, and only against the byte when it does not.
  // The packers have narrowed it to int with saturation (gemm_args.h's
  // narrow_to_int), so the refusals print it as narrowed.
  template <class Args>
  void sr_encode_args(Args &args, int variant, const char *op, CarrierBits c)
  {
    for_each_format_role(args, [&](const auto &widths, int n_fmt, auto &common, bool rounds) {
      common.prng_bits = rounds ? sr_encode(common.prng_bits, sr_widest_mantissa(widths, n_fmt), false, c, variant,
                                            op, true)
                                : sr_encode_narrowed(common.prng_bits, variant, op);
    });
  }

  // A GEMM call's Args as its kernel runs them: checks every format the
  // call rounds with, in every mode (format_host.h's check_args_formats), then
  // canonicalizes round_mode in place, and returns a copy of the Args whose
  // formats are encoded when the mode is stochastic. `c` is the carrier the
  // call rounds in.
  template <class Args>
  Args sr_canonical_args(const Args &args, int64_t &round_mode, const char *op, CarrierBits c)
  {
    check_args_formats(args, c, op);
    int variant;
    round_mode = sr_canonical(round_mode, variant);
    Args out = args;
    if (variant >= 0)
      sr_encode_args(out, variant, op, c);
    return out;
  }
  // The same for a block format's descriptor (its int[] fields in
  // block_decode.h's BlockFmtField order; `prng_field` is ELEM_PRNG_BITS):
  // canonicalizes round_mode in place, and encodes the element format's
  // prng_bits, which checked_block_fmt has already validated.
  template <class Fmt>
  void sr_canonical_fmt(Fmt &f, int prng_field, int64_t &round_mode, const char *op)
  {
    int variant;
    round_mode = sr_canonical(round_mode, variant);
    if (variant >= 0)
      f[prng_field] = sr_encode(f[prng_field], variant, op);
  }
} // namespace mptorch
