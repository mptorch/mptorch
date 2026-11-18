#pragma once

// The op boundary's checks of a format, made in every rounding mode, before
// anything is built from it.
//
// The parameter builders (make_binaryK_params, make_superfp_params and
// make_fixedpoint_params) shift by a format's widths, and compute exponents
// from its bias (a fixed-point format's from its fl) in an int. A width
// outside the range they were written for shifts by a negative count or by
// more than an int holds, and a bias far enough from zero overflows the int:
// both are undefined behaviour. mptorch.number refuses every such format
// before a call reaches an op, so these checks are what a direct torch.ops
// call meets. They are no stricter than the Python layer's, so no call it
// makes is refused here.

#include "bit_helper.h"
#include <c10/util/Exception.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <ostream>

namespace mptorch
{
  // The carrier a call rounds in, by its name and its stored mantissa bits:
  // binary64 for float64 tensors, binary32 for float32, float16 and bfloat16.
  struct CarrierBits
  {
    const char *name;
    int64_t man_bits;
  };

  inline CarrierBits carrier_bits(bool binary64)
  {
    return binary64 ? CarrierBits{"binary64", FloatTraits<double>::MAN_BITS}
                    : CarrierBits{"binary32", FloatTraits<float>::MAN_BITS};
  }

  // The widest exponent field the builders take: they compute 1 << exp_bits
  // in an int. The Python layer's bound is the same (_MAX_EXP_BITS in
  // mptorch/number/_carrier.py).
  constexpr int64_t MAX_EXP_BITS = 30;

  // The bound on the exponents the builders compute from a format's bias (or
  // from a fixed-point format's fl): they hold them in an int, and the casts
  // add the carrier's exponents and exponent bias to them, which are below
  // 1100 in magnitude. Every such exponent between -2^30 and 2^30 leaves
  // room for that. The Python layer's bound is the same (_EXP_LIMIT in
  // mptorch/number/_carrier.py).
  constexpr int64_t EXP_LIMIT = int64_t(1) << 30;

  enum class FormatFamily
  {
    BinaryK,
    Superfp,
    FixedPoint
  };

  // A format's parameters, as the checks read them. `bits` is a binaryK or
  // superfp format's stored mantissa bits, or a fixed-point format's
  // magnitude bits (its word length less the sign bit, if it has one). The
  // carrier stores the leading one of a fixed-point value's magnitude bits
  // implicitly, so it holds one more of them than mantissa bits. The two
  // modes are the schema's integers for SaturationMode and SubnormalsMode
  // (modes.h): both are binaryK's, only the first is superfp's, and fixed
  // point has neither.
  //
  // The quantizers pass the schema's int64 values. A GEMM's packers have
  // already narrowed its values to int, saturating a value past int's range
  // to INT_MIN or INT_MAX (gemm_args.h's narrow_to_int), and set `narrowed`:
  // every check refuses those limits, and a message prints a value at one of
  // them as "or less" or "or more" (Shown, below).
  struct Format
  {
    FormatFamily family;
    int64_t bits;
    int64_t exp_bits;        // binaryK and superfp
    int64_t bias;            // binaryK and superfp
    int64_t normal_binades;  // superfp
    int64_t fl;              // fixed point: the last bit stands for 2^-fl
    int64_t saturation_mode; // binaryK and superfp
    int64_t subnormals_mode; // binaryK
    bool narrowed = false;
  };

  // Narrows a value computed in 128 bits to int64, saturating: a value past
  // int64's range becomes INT64_MIN or INT64_MAX, which every check refuses
  // and a message prints as "or less" or "or more" (Shown, below).
  inline int64_t narrow_to_int64(__int128 v)
  {
    return v < INT64_MIN ? INT64_MIN : v > INT64_MAX ? INT64_MAX : static_cast<int64_t>(v);
  }

  // A binaryK format as a quantizer's schema gives it: P3109's (K, P), K
  // bits in all and P of precision, which make P - 1 stored mantissa bits
  // and an exponent field of the bits left after them and, in a signed
  // format, the sign bit. The two are computed in 128 bits, since for a K
  // or P near int64's limits the differences overflow an int64.
  inline Format binaryK_format(int64_t K, int64_t P, int64_t bias, bool is_signed, int64_t saturation_mode,
                               int64_t subnormals_mode)
  {
    const __int128 k = K, p = P;
    return Format{FormatFamily::BinaryK, narrow_to_int64(p - 1),
                  narrow_to_int64(is_signed ? k - p : k - p + 1), bias, 0, 0, saturation_mode, subnormals_mode};
  }

  inline Format superfp_format(int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias,
                               int64_t saturation_mode)
  {
    return Format{FormatFamily::Superfp, man_bits, exp_bits, bias, normal_binades, 0, saturation_mode, 0};
  }

  inline Format fixedpoint_format(int64_t mag_bits, int64_t fl)
  {
    return Format{FormatFamily::FixedPoint, mag_bits, 0, 0, 0, fl, 0, 0};
  }

  // A parameter as a refusal prints it. A value narrowed with saturation
  // that sits at a limit is that limit or any value past it, so it is
  // printed as "2147483647 or more" or "-2147483648 or less": a GEMM's value
  // at int's limits (`narrowed`), and any value at int64's, which only a
  // saturated difference (binaryK_format above) or the schema's own limit
  // can be.
  struct Shown
  {
    int64_t value;
    bool narrowed;
  };

  inline std::ostream &operator<<(std::ostream &o, Shown s)
  {
    o << s.value;
    if (s.value == INT64_MAX || (s.narrowed && s.value >= INT_MAX))
      o << " or more";
    else if (s.value == INT64_MIN || (s.narrowed && s.value <= INT_MIN))
      o << " or less";
    return o;
  }

  // How many bits of the carrier's significand a format's bits may take:
  // its stored mantissa bits, or one more for a fixed-point format's
  // magnitude bits.
  inline int64_t significand_room(bool magnitude, CarrierBits c) { return c.man_bits + (magnitude ? 1 : 0); }

  // Refuses a format the builders cannot take, checking, in this order:
  //   * its widths: an exponent field from 0 to MAX_EXP_BITS bits, and no
  //     more mantissa or magnitude bits than the carrier's significand holds
  //     (at least 0 mantissa bits, and at least 1 magnitude bit);
  //   * the exponents the builders compute from its bias, from 1 - bias (the
  //     exponent of field 1) to 2^exp_bits - 1 - bias (that of the top
  //     field), or from a fixed-point format's fl, from -fl (its last bit's)
  //     to mag_bits - fl (that of the binade above its top bit): each from
  //     -EXP_LIMIT to EXP_LIMIT;
  //   * a superfp format's normal_binades, from 1 to 2^exp_bits - 1, the
  //     Python layer's bound;
  //   * the saturation mode (binaryK and superfp) and the subnormals mode
  //     (binaryK), each a value of its enum in modes.h: the casts compare a
  //     mode with the members, so an integer that names none would run as
  //     whichever mode those comparisons select.
  // Each bound is compared with the parameter itself, so the check computes
  // nothing from a parameter that could overflow an int64. Defined out of
  // line, in format_checks.cpp, like the other checks of this header and of
  // stochastic_host.h. As inline functions compiled into each caller's
  // translation unit, edits to them changed the code GCC generated for the
  // quantizer loops beside them: the superfp quantizer's single-thread loop
  // ran from 3% slower to 7% faster, by dtype and mode (2026-11-18). Out of
  // line, those loops compile to the bytes they did before the checks.
  void check_format(const Format &f, CarrierBits c, const char *op);

  // Calls f(widths, n_fmt, common, rounds) for each format role of a GEMM's
  // Args: the multiply, accumulate and fused formats of the plain, palette
  // and cross-family Args (mul/mul_c, acc/acc_c, fma/fma_c), the outer
  // format of the accumulated ones (outer/outer_c), and, recursively, those
  // of the Args the odd, accumulated, conv and block wrappers hold (`base`,
  // `inner`). `widths` is one format, or a palette op's array of n_fmt
  // slots. `rounds` says whether the call rounds with the role's formats,
  // which it does unless the role's flag says otherwise: accumulate_quant,
  // fma_quant and outer_quant. The cross-family Args and the fused palette
  // Args have no flag, and always round. A format the call does not round
  // with is never built, and the Python layer passes it unchecked (an
  // accumulate format under accumulate_quant=false has P = 0, for example).
  template <class Args, class F>
  void for_each_format_role(Args &args, F &&f)
  {
    int n_fmt = 1;
    if constexpr (requires { args.n_fmt; })
      n_fmt = args.n_fmt;
    if constexpr (requires { args.mul_c; })
      f(args.mul, n_fmt, args.mul_c, true);
    if constexpr (requires { args.acc_c; })
    {
      bool rounds = true;
      if constexpr (requires { args.accumulate_quant; })
        rounds = args.accumulate_quant;
      f(args.acc, n_fmt, args.acc_c, rounds);
    }
    if constexpr (requires { args.fma_c; })
    {
      bool rounds = true;
      if constexpr (requires { args.fma_quant; })
        rounds = args.fma_quant;
      f(args.fma, n_fmt, args.fma_c, rounds);
    }
    if constexpr (requires { args.outer_c; })
      f(args.outer, 1, args.outer_c, args.outer_quant);
    if constexpr (requires { args.base; })
      for_each_format_role(args.base, f);
    if constexpr (requires { args.inner; })
      for_each_format_role(args.inner, f);
  }

  // Calls g on the role's one format, or on each of a palette's first n_fmt
  // slots.
  template <class Widths, class G>
  void for_each_slot(const Widths &w, int, G &&g)
  {
    g(w);
  }

  template <class Widths, size_t N, class G>
  void for_each_slot(const Widths (&w)[N], int n_fmt, G &&g)
  {
    for (int i = 0; i < n_fmt && i < static_cast<int>(N); ++i)
      g(w[i]);
  }

  // Checks every format a GEMM call rounds with (check_format), in the
  // carrier it rounds in. A role's widths are a BinaryKWidths or a
  // SuperfpWidths, and its modes those of its BinaryKCommon or SuperfpCommon
  // (gemm_args.h), all of which the packers have narrowed to int with
  // saturation (narrow_to_int), so each Format is marked narrowed.
  template <class Args>
  void check_args_formats(const Args &args, CarrierBits c, const char *op)
  {
    for_each_format_role(args, [&](const auto &widths, int n_fmt, const auto &common, bool rounds) {
      if (rounds)
        for_each_slot(widths, n_fmt, [&](const auto &w) {
          Format f{};
          // The family is read off common, whose type the outer lambda
          // fixes, so the other family's branch is discarded there. A test
          // on w stays dependent, and clang then substitutes common.sub
          // into the superfp instantiation, where it does not exist.
          if constexpr (requires { common.sub; })
            f = Format{FormatFamily::BinaryK, w.man_bits, w.exp_bits, w.bias, 0, 0,
                       static_cast<int64_t>(common.sat), static_cast<int64_t>(common.sub)};
          else
            f = superfp_format(w.man_bits, w.exp_bits, w.normal_binades, w.bias,
                               static_cast<int64_t>(common.sat));
          f.narrowed = true;
          check_format(f, c, op);
        });
    });
  }
} // namespace mptorch
