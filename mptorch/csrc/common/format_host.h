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
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstdint>

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

  // A format's parameters, as the checks read them: the schema's int64
  // values, before the builders narrow them to int. `bits` is a binaryK or
  // superfp format's stored mantissa bits, or a fixed-point format's
  // magnitude bits (its word length less the sign bit, if it has one). The
  // carrier stores the leading one of a fixed-point value's magnitude bits
  // implicitly, so it holds one more of them than mantissa bits.
  struct Format
  {
    FormatFamily family;
    int64_t bits;
    int64_t exp_bits;       // binaryK and superfp
    int64_t bias;           // binaryK and superfp
    int64_t normal_binades; // superfp
    int64_t fl;             // fixed point: the last bit stands for 2^-fl
  };

  inline Format binaryK_format(int64_t man_bits, int64_t exp_bits, int64_t bias)
  {
    return Format{FormatFamily::BinaryK, man_bits, exp_bits, bias, 0, 0};
  }

  inline Format superfp_format(int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias)
  {
    return Format{FormatFamily::Superfp, man_bits, exp_bits, bias, normal_binades, 0};
  }

  inline Format fixedpoint_format(int64_t mag_bits, int64_t fl)
  {
    return Format{FormatFamily::FixedPoint, mag_bits, 0, 0, 0, fl};
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
  //     Python layer's bound.
  // Each bound is compared with the parameter itself, so the check computes
  // nothing from a parameter that could overflow an int64. Not inlined: the
  // messages' code would grow each caller, and the compiler's inlining of
  // the callers' own code with it.
  C10_NOINLINE inline void check_format(const Format &f, CarrierBits c, const char *op)
  {
    const bool magnitude = f.family == FormatFamily::FixedPoint;
    const int64_t room = significand_room(magnitude, c);
    if (magnitude)
    {
      TORCH_CHECK(f.bits >= 1 && f.bits <= room, op, ": the format has ", f.bits, " magnitude bits, but ",
                  c.name, " holds 1 to ", room);
      TORCH_CHECK(f.fl >= f.bits - EXP_LIMIT && f.fl <= EXP_LIMIT, op, ": the format's fl is ", f.fl,
                  ", but with ", f.bits, " magnitude bits the kernels take ", f.bits - EXP_LIMIT, " to ",
                  EXP_LIMIT, ": they compute -fl and mag_bits - fl in an int, and need both between -2^30 and 2^30");
      return;
    }
    TORCH_CHECK(f.exp_bits >= 0 && f.exp_bits <= MAX_EXP_BITS, op, ": the format has ", f.exp_bits,
                " exponent bits, but the kernels take 0 to ", MAX_EXP_BITS, ": they compute 1 << exp_bits in an int");
    TORCH_CHECK(f.bits >= 0 && f.bits <= room, op, ": the format has ", f.bits, " stored mantissa bits, but ",
                c.name, " holds 0 to ", room);
    const int64_t top_field = (int64_t(1) << f.exp_bits) - 1;
    TORCH_CHECK(f.bias >= top_field - EXP_LIMIT && f.bias <= EXP_LIMIT + 1, op, ": the format's bias is ", f.bias,
                ", but with ", f.exp_bits, " exponent bits the kernels take ", top_field - EXP_LIMIT, " to ",
                EXP_LIMIT + 1,
                ": they compute 1 - bias and 2^exp_bits - 1 - bias in an int, and need both between -2^30 and 2^30");
    if (f.family == FormatFamily::Superfp)
      TORCH_CHECK(f.normal_binades >= 1 && f.normal_binades <= top_field, op, ": the format has ",
                  f.normal_binades, " normal binades, but a superfp format with ", f.exp_bits,
                  " exponent bits has 1 to ", top_field);
  }

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
  // SuperfpWidths (gemm_args.h), whose fields the packers have narrowed to
  // int.
  template <class Args>
  void check_args_formats(const Args &args, CarrierBits c, const char *op)
  {
    for_each_format_role(args, [&](const auto &widths, int n_fmt, const auto &, bool rounds) {
      if (rounds)
        for_each_slot(widths, n_fmt, [&](const auto &w) {
          if constexpr (requires { w.normal_binades; })
            check_format(superfp_format(w.man_bits, w.exp_bits, w.normal_binades, w.bias), c, op);
          else
            check_format(binaryK_format(w.man_bits, w.exp_bits, w.bias), c, op);
        });
    });
  }
} // namespace mptorch
