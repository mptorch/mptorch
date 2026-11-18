#pragma once

// The op boundary's checks of a format's widths, made in every rounding
// mode, before anything is built from them.
//
// The parameter builders (make_binaryK_params, make_superfp_params and
// make_fixedpoint_params) shift by a format's widths, so a width outside the
// range they were written for shifts by a negative count or by more than an
// int holds, which is undefined behaviour. mptorch.number refuses every such
// format before a call reaches an op, so these checks are what a direct
// torch.ops call meets. They are no stricter than the Python layer's, so no
// call it makes is refused here.

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

  // A format's widths, as the checks read them. A binaryK or superfp format
  // has `bits` stored mantissa bits and `exp_bits` exponent bits. A
  // fixed-point format has `bits` magnitude bits (its word length less the
  // sign bit, if it has one); the carrier stores the leading one of them
  // implicitly, so it holds one more magnitude bit than mantissa bits.
  struct FormatWidths
  {
    int64_t bits;
    int64_t exp_bits;
    bool magnitude;
  };

  inline FormatWidths float_widths(int64_t man_bits, int64_t exp_bits)
  {
    return FormatWidths{man_bits, exp_bits, false};
  }

  inline FormatWidths fixedpoint_widths(int64_t mag_bits) { return FormatWidths{mag_bits, 0, true}; }

  // How many bits of the carrier's significand a format's bits may take.
  inline int64_t significand_room(FormatWidths w, CarrierBits c) { return c.man_bits + (w.magnitude ? 1 : 0); }

  // Refuses a format whose widths the builders cannot take: an exponent
  // field outside [0, MAX_EXP_BITS], or more mantissa or magnitude bits than
  // the carrier's significand holds (at least 0 mantissa bits, and at least
  // 1 magnitude bit). Not inlined: the messages' code would grow each
  // caller, and the compiler's inlining of the callers' own code with it.
  C10_NOINLINE inline void check_format_widths(FormatWidths w, CarrierBits c, const char *op)
  {
    const int64_t room = significand_room(w, c);
    if (w.magnitude)
    {
      TORCH_CHECK(w.bits >= 1 && w.bits <= room, op, ": the format has ", w.bits, " magnitude bits, but ",
                  c.name, " holds 1 to ", room);
      return;
    }
    TORCH_CHECK(w.exp_bits >= 0 && w.exp_bits <= MAX_EXP_BITS, op, ": the format has ", w.exp_bits,
                " exponent bits, but the kernels take 0 to ", MAX_EXP_BITS, ": they compute 1 << exp_bits in an int");
    TORCH_CHECK(w.bits >= 0 && w.bits <= room, op, ": the format has ", w.bits, " stored mantissa bits, but ",
                c.name, " holds 0 to ", room);
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

  // Checks the widths of every format a GEMM call rounds with, in the
  // carrier it rounds in.
  template <class Args>
  void check_args_widths(const Args &args, CarrierBits c, const char *op)
  {
    for_each_format_role(args, [&](const auto &widths, int n_fmt, const auto &, bool rounds) {
      if (rounds)
        for_each_slot(widths, n_fmt, [&](const auto &w) {
          check_format_widths(float_widths(w.man_bits, w.exp_bits), c, op);
        });
    });
  }
} // namespace mptorch
