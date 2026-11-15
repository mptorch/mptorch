#pragma once

// The palette ops with the carrier step rounded to odd
// (custom_matmul_*_mixed_odd and custom_conv_*_mixed_odd;
// carrier_rounding=RoundMode.RO): OddPaletteArgs, a mixed Args whose palette
// holds the odd twins of the base's Macs, slot by slot.
//
// A header of its own, next to carrier_odd.h rather than in it, so that the
// objects that include carrier_odd.h (the single-format odd ops') stay what
// they were; only the palette ops' units include this one. The MPS backend
// compiles it as Metal too (mps/odd.metal, phase Z), where the kernel builds
// one element's slot at a time: there OddPaletteArgs has with_slot in place
// of with_palette.

#include "carrier_odd.h"
#include "gemm_args.h"

#if !defined(__METAL_VERSION__)
#include <type_traits>
#endif

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

    static constexpr MPTORCH_CONSTANT bool mixed = true;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = Base::draws_per_k_step;

    Base base{};
    int n_fmt = 0;

#if !defined(__METAL_VERSION__)
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
#else
    // The MPS kernel builds one element's Mac from its slot (the base's
    // with_slot, mps/gemm.metal says why) rather than a palette of them, and
    // OddArgs' Builder makes that Mac its odd twin, as it does a
    // single-format op's: the base hands it the NaiveAccumulator its slot's
    // widths build.
    template <class T, RoundMode RM, class F>
    void with_slot(int32_t idx, MPTORCH_THREAD F &&f) const
    {
      base.template with_slot<T, RM>(idx, typename OddArgs<Base>::template Builder<T, F>{f});
    }
#endif
  };

#if !defined(__METAL_VERSION__)
  template <class Base>
  OddPaletteArgs<Base> odd_palette(const Base &base)
  {
    OddPaletteArgs<Base> args;
    args.base = base;
    args.n_fmt = base.n_fmt;
    return args;
  }
#endif

} // namespace mptorch::gemm
