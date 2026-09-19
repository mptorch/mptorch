#pragma once

// One plain struct per GEMM op, holding exactly the parameters that op's
// `TORCH_LIBRARY` schema carries, plus the factory that turns them into the
// policy objects the kernel is templated on. This is the only part that
// genuinely differs between the eight ops; everything they share (the input
// checks, the shape derivation, the output allocation, the RNG draw) is the
// driver in common/gemm_host.h, and an entry point is one packer call plus
// one driver call.
//
// Deliberately ATen-free: a .cu names these structs, and including
// <ATen/core/Tensor.h> for that would cost about 22 s of nvcc per
// translation unit (see common/gemm_dtype.h). The format widths therefore
// arrive already derived. The binaryK schemas' (K, P) spelling is converted
// to (man_bits, exp_bits) by binaryK_widths in the host-side packer, so this
// header never has to know which spelling a schema used.
//
// with_accumulator<T, RM> and with_palette<T, RM> build the policies in
// carrier T (common/gemm_policy.h). The widths are the format's and do not
// depend on the carrier, so one packed Args serves both binary32 and
// binary64, and which carrier runs is the backend's choice, made from
// GemmShape::dt.

#include "gemm_policy.h"
#include "modes.h"
#if !defined(__METAL_VERSION__)
#include "gemm_dtype.h"
#include <cstdint>
#endif

namespace mptorch::gemm
{

#if !defined(__METAL_VERSION__)
  // Everything the kernel needs that is not a format: the operands as raw
  // pointers plus the shape, layout and dtype to read them with. The mixed
  // ops fill the last four fields; the single-format ops leave them null,
  // which is what selects the kernel's MIXED=false instantiation.
  //
  // The kernels carry exactly one batch dimension. `batch` is the broadcast
  // batch size, and the two operand strides are *element* offsets between
  // consecutive batch elements, where 0 means broadcast: a `[K, N]` weight
  // shared by every batch element rides at stride 0 rather than being
  // materialized `batch` times. C is always written densely as
  // [batch, M, N], so its stride is M*N and is not carried. A 2D call is
  // batch = 1 with both strides 0, which is the same arithmetic as before
  // batching existed: nothing downstream branches on it.
  struct GemmShape
  {
    const void *a = nullptr;
    const void *b = nullptr;
    void *c = nullptr;
    GemmDtype dt = GemmDtype::Float;
    int64_t M = 0, K = 0, N = 0;
    int64_t batch = 1;
    int64_t stride_a = 0, stride_b = 0;
    bool trans_a = false, trans_b = false;
    RoundMode rm = RoundMode::RNE;
    bool use_rng = false;
    const int32_t *prec_idx = nullptr;
    int64_t idx_row_stride = 0, idx_col_stride = 0, idx_batch_stride = 0;
  };
#endif

  // The per-format settings a schema takes as scalars rather than per slot:
  // sign, saturation, subnormals and the stochastic-rounding bit width. A
  // mixed op tabulates only the format *widths* per palette slot and shares
  // one of these across every slot.
  struct BinaryKCommon
  {
    bool is_signed = true;
    SaturationMode sat{};
    SubnormalsMode sub{};
    int prng_bits = 0;
  };

  struct SuperfpCommon
  {
    bool is_signed = true;
    SaturationMode sat{};
    int prng_bits = 0;
  };

  // One palette slot's widths. `n_fmt` of these ride alongside a *Common.
  struct BinaryKWidths
  {
    int man_bits = 0, exp_bits = 0, bias = 0;
  };

  struct SuperfpWidths
  {
    int man_bits = 0, exp_bits = 0, normal_binades = 0, bias = 0;
  };

  // The schemas spell a binaryK format as (K, P), total bits and precision,
  // the two parameters IEEE P3109 names its binaryK formats by, where the
  // policies want (man_bits, exp_bits). The precision counts the implicit
  // leading bit, so man_bits = P - 1; the exponent gets what is left after
  // the significand and, in a signed format, the sign bit.
  inline BinaryKWidths binaryK_widths(int64_t K, int64_t P, int64_t bias, bool is_signed)
  {
    return BinaryKWidths{static_cast<int>(P - 1),
                         static_cast<int>(is_signed ? K - P : K - P + 1),
                         static_cast<int>(bias)};
  }

  // A superfp format's widths, as the schema spells them, narrowed to int.
  inline SuperfpWidths superfp_widths(int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                                      int64_t bias)
  {
    return SuperfpWidths{static_cast<int>(man_bits), static_cast<int>(exp_bits),
                         static_cast<int>(normal_binades), static_cast<int>(bias)};
  }

  // The four policy constructors, in carrier T and rounding mode RM, from a
  // slot's widths and the op's shared settings. Each builds its cast
  // constants (BinaryKParamsT / SuperfpParamsT) once here rather than per
  // multiply-accumulate step.
  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE BinaryKMultiplierT<T, RM> make_mul(const MPTORCH_THREAD BinaryKWidths &w,
                                                             const MPTORCH_THREAD BinaryKCommon &c)
  {
    return BinaryKMultiplierT<T, RM>{w.man_bits, w.exp_bits, w.bias, c.is_signed, c.sat, c.sub, c.prng_bits};
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE BinaryKAdderT<T, RM> make_add(const MPTORCH_THREAD BinaryKWidths &w,
                                                        const MPTORCH_THREAD BinaryKCommon &c)
  {
    return BinaryKAdderT<T, RM>{w.man_bits, w.exp_bits, w.bias, c.is_signed, c.sat, c.sub, c.prng_bits};
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE SuperfpMultiplierT<T, RM> make_mul(const MPTORCH_THREAD SuperfpWidths &w,
                                                             const MPTORCH_THREAD SuperfpCommon &c)
  {
    return SuperfpMultiplierT<T, RM>{w.man_bits, w.exp_bits, w.normal_binades, w.bias, c.is_signed, c.sat,
                                  c.prng_bits};
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE SuperfpAdderT<T, RM> make_add(const MPTORCH_THREAD SuperfpWidths &w,
                                                        const MPTORCH_THREAD SuperfpCommon &c)
  {
    return SuperfpAdderT<T, RM>{w.man_bits, w.exp_bits, w.normal_binades, w.bias, c.is_signed, c.sat,
                             c.prng_bits};
  }

  // Each Args struct below carries two constants the driver and the kernel
  // read: `mixed`, which says whether the op takes a palette and a per-
  // element precision index, and `draws_per_k_step`, the number of random
  // values one output element may consume per K-step under RoundMode::SR.
  // Its `with_accumulator<T, RM>(f)` (single-format) or
  // `with_palette<T, RM>(f)` (mixed) builds the NaiveAccumulator prototype
  // in carrier T, and the palette of Macs on the mixed path, and hands them
  // to `f`, which launches the kernel. An `accumulate_quant` / `fma_quant`
  // of false substitutes IdentityAdder for the sum's cast, so the running
  // sum stays in the carrier's precision.

  // ----------------------------------------------------------------------
  // binaryK, split mac (custom_matmul_binaryK)
  // ----------------------------------------------------------------------
  struct BinaryKSplitArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    // SplitMac draws for the multiply and the accumulate independently, so
    // one output element's K-step reduction consumes two Philox values per
    // step where a FusedMac consumes one. Only RoundMode::SR draws at all;
    // this is the upper bound the driver reserves generator state against.
    // A draw is one word in binary32 and two in binary64, which the driver
    // multiplies in.
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 2;

    BinaryKWidths mul{};
    BinaryKCommon mul_c{};
    bool accumulate_quant = false;
    BinaryKWidths acc{};
    BinaryKCommon acc_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      auto m = make_mul<T, RM>(mul, mul_c);
      if (accumulate_quant)
      {
        using Mac = SplitMac<BinaryKMultiplierT<T, RM>, BinaryKAdderT<T, RM>>;
        f(NaiveAccumulator<Mac>{Mac{m, make_add<T, RM>(acc, acc_c)}, T(0)});
      }
      else
      {
        using Mac = SplitMac<BinaryKMultiplierT<T, RM>, IdentityAdder<T>>;
        f(NaiveAccumulator<Mac>{Mac{m, IdentityAdder<T>{}}, T(0)});
      }
    }
  };

  struct BinaryKSplitMixedArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = true;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 2;

    int n_fmt = 0;
    BinaryKWidths mul[MAX_GEMM_FORMATS]{};
    BinaryKCommon mul_c{};
    bool accumulate_quant = false;
    BinaryKWidths acc[MAX_GEMM_FORMATS]{};
    BinaryKCommon acc_c{};

    // Every slot is a fully built Mac, cast constants included, so the
    // kernel's per-element prologue copies one and the K-loop rebuilds
    // nothing. The prototype handed to `f` holds slot 0; the kernel replaces
    // its Mac per output element from `pal`.
    template <class T, RoundMode RM, class F>
    void with_palette(MPTORCH_THREAD F &&f) const
    {
      if (accumulate_quant)
      {
        using Mac = SplitMac<BinaryKMultiplierT<T, RM>, BinaryKAdderT<T, RM>>;
        FormatPalette<Mac> pal;
        for (int i = 0; i < n_fmt; ++i)
          pal.slots[i] = Mac{make_mul<T, RM>(mul[i], mul_c), make_add<T, RM>(acc[i], acc_c)};
        f(NaiveAccumulator<Mac>{pal.slots[0], T(0)}, pal);
      }
      else
      {
        using Mac = SplitMac<BinaryKMultiplierT<T, RM>, IdentityAdder<T>>;
        FormatPalette<Mac> pal;
        for (int i = 0; i < n_fmt; ++i)
          pal.slots[i] = Mac{make_mul<T, RM>(mul[i], mul_c), IdentityAdder<T>{}};
        f(NaiveAccumulator<Mac>{pal.slots[0], T(0)}, pal);
      }
    }
  };

  // ----------------------------------------------------------------------
  // binaryK, fused mac (custom_matmul_binaryK_fma)
  // ----------------------------------------------------------------------
  struct BinaryKFusedArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 1;

    bool fma_quant = false;
    BinaryKWidths fma{};
    BinaryKCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      if (fma_quant)
      {
        using Mac = FusedMac<BinaryKAdderT<T, RM>>;
        f(NaiveAccumulator<Mac>{Mac{make_add<T, RM>(fma, fma_c)}, T(0)});
      }
      else
      {
        using Mac = FusedMac<IdentityAdder<T>>;
        f(NaiveAccumulator<Mac>{Mac{IdentityAdder<T>{}}, T(0)});
      }
    }
  };

  // fma_quant=false has no counterpart here: FusedMac<IdentityAdder> carries
  // no format, so a palette of it would make prec_idx a no-op. The entry
  // point rejects it.
  struct BinaryKFusedMixedArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = true;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 1;

    int n_fmt = 0;
    BinaryKWidths fma[MAX_GEMM_FORMATS]{};
    BinaryKCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_palette(MPTORCH_THREAD F &&f) const
    {
      using Mac = FusedMac<BinaryKAdderT<T, RM>>;
      FormatPalette<Mac> pal;
      for (int i = 0; i < n_fmt; ++i)
        pal.slots[i] = Mac{make_add<T, RM>(fma[i], fma_c)};
      f(NaiveAccumulator<Mac>{pal.slots[0], T(0)}, pal);
    }
  };

  // ----------------------------------------------------------------------
  // superfp, split mac (custom_matmul_superfp)
  // ----------------------------------------------------------------------
  struct SuperfpSplitArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 2;

    SuperfpWidths mul{};
    SuperfpCommon mul_c{};
    bool accumulate_quant = false;
    SuperfpWidths acc{};
    SuperfpCommon acc_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      auto m = make_mul<T, RM>(mul, mul_c);
      if (accumulate_quant)
      {
        using Mac = SplitMac<SuperfpMultiplierT<T, RM>, SuperfpAdderT<T, RM>>;
        f(NaiveAccumulator<Mac>{Mac{m, make_add<T, RM>(acc, acc_c)}, T(0)});
      }
      else
      {
        using Mac = SplitMac<SuperfpMultiplierT<T, RM>, IdentityAdder<T>>;
        f(NaiveAccumulator<Mac>{Mac{m, IdentityAdder<T>{}}, T(0)});
      }
    }
  };

  struct SuperfpSplitMixedArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = true;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 2;

    int n_fmt = 0;
    SuperfpWidths mul[MAX_GEMM_FORMATS]{};
    SuperfpCommon mul_c{};
    bool accumulate_quant = false;
    SuperfpWidths acc[MAX_GEMM_FORMATS]{};
    SuperfpCommon acc_c{};

    // Two full superfp policies per slot make this the largest palette of
    // the eight. It fits in registers only because the rounding mode is a
    // template parameter: with a runtime switch over seven cast bodies this
    // kernel needed 100 registers, and fixing the mode at compile time took
    // it to 60-64.
    template <class T, RoundMode RM, class F>
    void with_palette(MPTORCH_THREAD F &&f) const
    {
      if (accumulate_quant)
      {
        using Mac = SplitMac<SuperfpMultiplierT<T, RM>, SuperfpAdderT<T, RM>>;
        FormatPalette<Mac> pal;
        for (int i = 0; i < n_fmt; ++i)
          pal.slots[i] = Mac{make_mul<T, RM>(mul[i], mul_c), make_add<T, RM>(acc[i], acc_c)};
        f(NaiveAccumulator<Mac>{pal.slots[0], T(0)}, pal);
      }
      else
      {
        using Mac = SplitMac<SuperfpMultiplierT<T, RM>, IdentityAdder<T>>;
        FormatPalette<Mac> pal;
        for (int i = 0; i < n_fmt; ++i)
          pal.slots[i] = Mac{make_mul<T, RM>(mul[i], mul_c), IdentityAdder<T>{}};
        f(NaiveAccumulator<Mac>{pal.slots[0], T(0)}, pal);
      }
    }
  };

  // ----------------------------------------------------------------------
  // superfp, fused mac (custom_matmul_superfp_fma)
  // ----------------------------------------------------------------------
  struct SuperfpFusedArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 1;

    bool fma_quant = false;
    SuperfpWidths fma{};
    SuperfpCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      if (fma_quant)
      {
        using Mac = FusedMac<SuperfpAdderT<T, RM>>;
        f(NaiveAccumulator<Mac>{Mac{make_add<T, RM>(fma, fma_c)}, T(0)});
      }
      else
      {
        using Mac = FusedMac<IdentityAdder<T>>;
        f(NaiveAccumulator<Mac>{Mac{IdentityAdder<T>{}}, T(0)});
      }
    }
  };

  // Like BinaryKFusedMixedArgs, this has no fma_quant=false form; the entry
  // point rejects it.
  struct SuperfpFusedMixedArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = true;
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 1;

    int n_fmt = 0;
    SuperfpWidths fma[MAX_GEMM_FORMATS]{};
    SuperfpCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_palette(MPTORCH_THREAD F &&f) const
    {
      using Mac = FusedMac<SuperfpAdderT<T, RM>>;
      FormatPalette<Mac> pal;
      for (int i = 0; i < n_fmt; ++i)
        pal.slots[i] = Mac{make_add<T, RM>(fma[i], fma_c)};
      f(NaiveAccumulator<Mac>{pal.slots[0], T(0)}, pal);
    }
  };

} // namespace mptorch::gemm
