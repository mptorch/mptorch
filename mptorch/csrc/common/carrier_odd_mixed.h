#pragma once

// The palette ops with the carrier step rounded to odd
// (custom_matmul_*_mixed_odd and custom_conv_*_mixed_odd;
// carrier_rounding=RoundMode.RO): OddPaletteArgs, a mixed Args whose palette
// holds the odd twins of the base's Macs, slot by slot.
//
// A header of its own, next to carrier_odd.h rather than in it, so that the
// objects that include carrier_odd.h (the single-format odd ops') stay what
// they were; only the palette ops' units include this one.

#include "carrier_odd.h"
#include "gemm_args.h"

#include <type_traits>

#if !defined(__METAL_VERSION__)

namespace mptorch::gemm
{
  // An existing palette Args with each slot's Mac rebuilt as its odd twin,
  // the way OddArgs rebuilds a single-format one: the base's with_palette
  // builds the palette of plain Macs, cast constants and all, and
  // PaletteBuilder copies each slot into a FusedMacOdd or a SplitMacOdd,
  // which hold the same policies. The kernels take the result unchanged, a
  // FormatPalette of the odd Mac, which CUDA passes by value as a kernel
  // parameter as it does the plain one.
  //
  // A wrapper with a `base` member, like OddArgs and AccumulateArgs, rather
  // than a class derived from the mixed Args: sr_canonical_args and
  // sr_encode_args reach the formats through `.base`, and nothing at the top
  // level may be named like a base field (mul_c, acc_c, fma_c, outer_c), an
  // inner Args (`inner`) or an algorithm (`alg`), or the prng bits would be
  // encoded twice and launch_conv would take its accumulated branch. The
  // drivers read `n_fmt` as a member, so the wrapper carries its own copy,
  // which odd_palette() sets from the base and nothing else writes.
  template <class Base>
  struct OddPaletteArgs
  {
    static_assert(Base::mixed, "OddPaletteArgs wraps a palette Args; OddArgs wraps the others");

    static constexpr bool mixed = true;
    static constexpr uint64_t draws_per_k_step = Base::draws_per_k_step;

    Base base{};
    int n_fmt = 0;

    template <class T, class F>
    struct PaletteBuilder
    {
      F &f;

      template <class Adder>
      void operator()(const NaiveAccumulator<FusedMac<Adder>> &,
                      const FormatPalette<FusedMac<Adder>> &pal) const
      {
        // The fused palette ops refuse fma_quant = false before packing: a
        // fused step that no format rounds has no format to vary.
        static_assert(!is_identity_adder<Adder>::value,
                      "a fused palette always rounds its step to a format");
        using Mac = FusedMacOdd<Adder>;
        static_assert(sizeof(Mac) == sizeof(FusedMac<Adder>) &&
                          std::is_trivially_copyable_v<Mac>,
                      "an odd Mac holds what its plain Mac holds");
        FormatPalette<Mac> odd;
        for (int i = 0; i < MAX_GEMM_FORMATS; ++i)
          odd.slots[i] = Mac{pal.slots[i].add};
        f(NaiveAccumulator<Mac>{odd.slots[0], T(0)}, odd);
      }

      // Over an IdentityAdder (accumulate_quant = false) only the product is
      // rounded to odd, as SplitMacOdd states.
      template <class Multiplier, class Adder>
      void operator()(const NaiveAccumulator<SplitMac<Multiplier, Adder>> &,
                      const FormatPalette<SplitMac<Multiplier, Adder>> &pal) const
      {
        using Mac = SplitMacOdd<Multiplier, Adder>;
        static_assert(sizeof(Mac) == sizeof(SplitMac<Multiplier, Adder>) &&
                          std::is_trivially_copyable_v<Mac>,
                      "an odd Mac holds what its plain Mac holds");
        FormatPalette<Mac> odd;
        for (int i = 0; i < MAX_GEMM_FORMATS; ++i)
          odd.slots[i] = Mac{pal.slots[i].mul, pal.slots[i].add};
        f(NaiveAccumulator<Mac>{odd.slots[0], T(0)}, odd);
      }
    };

    template <class T, RoundMode RM, class F>
    void with_palette(F &&f) const
    {
      base.template with_palette<T, RM>(PaletteBuilder<T, F>{f});
    }
  };

  template <class Base>
  OddPaletteArgs<Base> odd_palette(const Base &base)
  {
    OddPaletteArgs<Base> args;
    args.base = base;
    args.n_fmt = base.n_fmt;
    return args;
  }

} // namespace mptorch::gemm

#endif // !__METAL_VERSION__
