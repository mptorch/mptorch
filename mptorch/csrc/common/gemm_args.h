#pragma once

// One plain struct (an "Args") for each of the eight NAIVE GEMM ops
// (custom_matmul_{binaryK,superfp}{,_fma}{,_mixed}), holding exactly the
// parameters of that op's TORCH_LIBRARY schema, plus the factory that turns
// them into the policy objects the kernel is a template over. The Args of
// the other GEMM-shaped ops wrap these: AccumulateArgs (gemm_accumulate.h),
// ConvArgs (gemm_gather.h), OddArgs and OddPaletteArgs (carrier_odd.h,
// carrier_odd_mixed.h); the cross-family and block Args are in gemm_cross.h
// and gemm_block.h. The Args are the only part that differs between the
// ops. Everything they share (the input checks, the shape derivation, the
// output allocation, the random-state draw) is in the drivers of
// gemm_host.h, so an entry point is one packer call plus one driver call.
//
// This header deliberately uses no ATen: the .cu files name these structs,
// and including <ATen/core/Tensor.h> for that would cost about 22 s of nvcc
// per translation unit (see gemm_dtype.h). So the format widths arrive
// already derived: the host-side packer converts the binaryK schemas'
// (K, P) to (man_bits, exp_bits) with binaryK_widths, and this header never
// sees which form a schema used.
//
// with_accumulator<T, RM> and with_palette<T, RM> build the policies in the
// carrier T (gemm_policy.h). The widths belong to the format and do not
// depend on the carrier, so one packed Args serves both binary32 and
// binary64, and the backend chooses the carrier from GemmShape::dt.

#include "gemm_policy.h"
#include "modes.h"
#if !defined(__METAL_VERSION__)
#include "gemm_dtype.h"
#include <climits>
#include <cstdint>
#endif

namespace mptorch::gemm
{

#if !defined(__METAL_VERSION__)
  // Everything the kernel needs besides the formats: the operands as raw
  // pointers, and the shape, layout and dtype to read them with. The palette
  // ops fill in the last four fields; the single-format ops leave them null
  // and zero, and run the kernel's MIXED=false instantiation.
  //
  // The kernels have exactly one batch dimension. `batch` is the batch size
  // after broadcasting, and the two operand strides are offsets, in elements,
  // between consecutive batch elements, where 0 means that every batch
  // element reads the same matrix: a `[K, N]` weight shared by the whole
  // batch has stride 0 rather than being copied `batch` times. C is always
  // written densely as [batch, M, N], so its stride is M*N and is not stored.
  // A 2D call is batch = 1 with both strides 0, so the kernels need no
  // separate 2D path.
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

  // The settings of a format that a schema takes as single values rather
  // than one per palette slot: the sign, the saturation and subnormal modes,
  // and the number of random bits stochastic rounding uses. A palette op
  // lists only the format *widths* per slot, and shares one of these between
  // all its slots.
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

  // One palette slot's widths. A palette op has `n_fmt` of these and one
  // *Common.
  struct BinaryKWidths
  {
    int man_bits = 0, exp_bits = 0, bias = 0;
  };

  struct SuperfpWidths
  {
    int man_bits = 0, exp_bits = 0, normal_binades = 0, bias = 0;
  };

#if !defined(__METAL_VERSION__)
  // Narrows a schema integer to int for the packers, saturating: a value
  // past int's range becomes INT_MIN or INT_MAX. Every check of a format
  // (format_host.h's check_format, stochastic_host.h's sr_encode) refuses
  // those, whereas a plain conversion would wrap the value to one a check
  // could accept: a bias of 2^32 + 7 would become 7. The checks therefore
  // print a narrowed value at int's limit as "or more" or "or less". It
  // takes a __int128 so that binaryK_widths can narrow K - P computed
  // exactly, since that difference can overflow an int64.
  inline int narrow_to_int(__int128 v) { return v < INT_MIN ? INT_MIN : v > INT_MAX ? INT_MAX : static_cast<int>(v); }

  // The schemas give a binaryK format as (K, P), its total number of bits
  // and its precision, the two parameters by which IEEE P3109 names its
  // binaryK formats, while the policies take (man_bits, exp_bits). The
  // precision counts the implicit leading bit, so man_bits = P - 1, and the
  // exponent gets the bits left after the significand and, in a signed
  // format, the sign bit. Both are computed in 128 bits and narrowed with
  // narrow_to_int, as is the bias.
  inline BinaryKWidths binaryK_widths(int64_t K, int64_t P, int64_t bias, bool is_signed)
  {
    const __int128 k = K, p = P;
    return BinaryKWidths{narrow_to_int(p - 1), narrow_to_int(is_signed ? k - p : k - p + 1), narrow_to_int(bias)};
  }

  // A superfp format's widths, as the schema gives them, narrowed to int
  // with narrow_to_int.
  inline SuperfpWidths superfp_widths(int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                                      int64_t bias)
  {
    return SuperfpWidths{narrow_to_int(man_bits), narrow_to_int(exp_bits), narrow_to_int(normal_binades),
                         narrow_to_int(bias)};
  }
#endif

  // The four policy constructors, in carrier T and rounding mode RM, from a
  // slot's widths and the op's shared settings. Each policy computes its
  // cast's constants (BinaryKParamsT or SuperfpParamsT) once, here, rather
  // than on every multiply-accumulate step.
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

  // Each Args struct below has two constants that the driver and the kernel
  // read: `mixed`, which says whether the op takes a palette and a precision
  // index per output element, and `draws_per_k_step`, the largest number of
  // random draws one output element may take per K-step under
  // RoundMode::SR. Its `with_accumulator<T, RM>(f)` (single-format) or
  // `with_palette<T, RM>(f)` (palette) builds the NaiveAccumulator prototype
  // in carrier T, plus the palette of Macs on the palette path, and passes
  // them to `f`, which launches the kernel. When `accumulate_quant` or
  // `fma_quant` is false, an IdentityAdder replaces the sum's cast, so the
  // running sum stays in the carrier's precision.
  //
  // The palette Args also have `with_slot<T, RM>(idx, f)`, which builds only
  // the Mac that precision index `idx` selects, from that slot's widths, and
  // passes `f` an accumulator holding it. The MPS kernel calls it once per
  // output element instead of with_palette (the comments in mps/gemm.metal
  // explain why); the Mac it builds is the one with_palette puts in that
  // slot.

  // Entry `idx` of a palette's per-slot list. The index is masked as
  // FormatPalette::slot masks it, and a masked index at or above `n_fmt`
  // gives entry 0, one of the op's formats, where FormatPalette::slot would
  // give an empty slot; the host's check of the map rules out both cases.
  // The entry is found by comparing rather than by indexing, and with_slot
  // relies on that: a list indexed at run time has to be kept in memory, and
  // every use of the Mac built from its entry then becomes a load.
  template <class W>
  CUDA_HOST_DEVICE_INLINE W palette_entry(const MPTORCH_THREAD W *list, int n_fmt, int32_t idx)
  {
    const int32_t s = idx & (MAX_GEMM_FORMATS - 1);
    W w = list[0];
    for (int k = 1; k < n_fmt; ++k)
      if (s == k)
        w = list[k];
    return w;
  }

  // ----------------------------------------------------------------------
  // binaryK, split mac (custom_matmul_binaryK)
  // ----------------------------------------------------------------------
  struct BinaryKSplitArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    // A SplitMac draws once for the multiply and once for the add, so one
    // K-step of one output element takes two random draws where a FusedMac
    // takes one. Only RoundMode::SR draws at all; this is the upper bound for
    // which the driver reserves generator state. A draw is one Philox word in
    // binary32 and two in binary64, a factor the driver applies.
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

    // Every slot is a fully built Mac, its cast constants included, so the
    // kernel's per-element prologue takes one and the K-loop builds nothing.
    // The prototype passed to `f` holds slot 0; the kernel replaces its Mac
    // for each output element from `pal`.
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

    template <class T, RoundMode RM, class F>
    void with_slot(int32_t idx, MPTORCH_THREAD F &&f) const
    {
      const auto m = make_mul<T, RM>(palette_entry(mul, n_fmt, idx), mul_c);
      if (accumulate_quant)
      {
        using Mac = SplitMac<BinaryKMultiplierT<T, RM>, BinaryKAdderT<T, RM>>;
        f(NaiveAccumulator<Mac>{Mac{m, make_add<T, RM>(palette_entry(acc, n_fmt, idx), acc_c)},
                                T(0)});
      }
      else
      {
        using Mac = SplitMac<BinaryKMultiplierT<T, RM>, IdentityAdder<T>>;
        f(NaiveAccumulator<Mac>{Mac{m, IdentityAdder<T>{}}, T(0)});
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

  // This has no fma_quant=false form: a FusedMac<IdentityAdder> rounds to no
  // format, so a palette of them would make prec_idx do nothing. The entry
  // point rejects fma_quant=false.
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

    template <class T, RoundMode RM, class F>
    void with_slot(int32_t idx, MPTORCH_THREAD F &&f) const
    {
      using Mac = FusedMac<BinaryKAdderT<T, RM>>;
      f(NaiveAccumulator<Mac>{Mac{make_add<T, RM>(palette_entry(fma, n_fmt, idx), fma_c)}, T(0)});
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

    // Two full superfp policies per slot make this the largest of the four
    // palettes. Its kernel fits in registers only because the rounding mode
    // is a template parameter: with a run-time switch over seven cast bodies
    // it needed 100 registers, and fixing the mode at compile time took it to
    // 60-64.
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

    template <class T, RoundMode RM, class F>
    void with_slot(int32_t idx, MPTORCH_THREAD F &&f) const
    {
      const auto m = make_mul<T, RM>(palette_entry(mul, n_fmt, idx), mul_c);
      if (accumulate_quant)
      {
        using Mac = SplitMac<SuperfpMultiplierT<T, RM>, SuperfpAdderT<T, RM>>;
        f(NaiveAccumulator<Mac>{Mac{m, make_add<T, RM>(palette_entry(acc, n_fmt, idx), acc_c)},
                                T(0)});
      }
      else
      {
        using Mac = SplitMac<SuperfpMultiplierT<T, RM>, IdentityAdder<T>>;
        f(NaiveAccumulator<Mac>{Mac{m, IdentityAdder<T>{}}, T(0)});
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

  // Like BinaryKFusedMixedArgs, this has no fma_quant=false form, and the
  // entry point rejects fma_quant=false.
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

    template <class T, RoundMode RM, class F>
    void with_slot(int32_t idx, MPTORCH_THREAD F &&f) const
    {
      using Mac = FusedMac<SuperfpAdderT<T, RM>>;
      f(NaiveAccumulator<Mac>{Mac{make_add<T, RM>(palette_entry(fma, n_fmt, idx), fma_c)}, T(0)});
    }
  };

} // namespace mptorch::gemm
