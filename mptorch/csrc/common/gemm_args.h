#pragma once

// One plain struct per GEMM op, holding exactly the parameters that op's
// `TORCH_LIBRARY` schema carries, plus the factory that turns them into the
// policy objects the kernel is templated on.
//
// Before H1 this lived sixteen times: each entry point derived its own
// man/exp bits, built its own Mac inside its own `dispatch_round_mode`
// lambda, and handed it to its own launch call, wrapped in ~60 lines of
// prologue identical to the other fifteen. The prologue is now
// common/gemm_host.h; what is left here is the only part that genuinely
// differs between the eight ops, expressed once and shared by both backends.
//
// Deliberately ATen-free (see common/gemm_dtype.h for why): a .cu names these
// structs, and paying <ATen/core/Tensor.h> for that would cost ~22 s of nvcc
// per translation unit. The format widths therefore arrive already derived --
// binaryK's (K, P) -> (man_bits, exp_bits) convention is applied by the entry
// point in the .cpp, where the schema is bound -- so this header never has to
// know which spelling a schema used.
//
// with_accumulator<T, RM> and with_palette<T, RM> build the policies in
// carrier T (common/gemm_policy.h): the widths are the format's and do not
// depend on it, so one packed Args serves both carriers, and which one runs is
// the backend's choice, made on GemmShape::dt.

#include "gemm_dtype.h"
#include "gemm_policy.h"
#include "modes.h"
#include <cstdint>

namespace mptorch::gemm
{

  // Everything the kernel needs that is not a format: the operands as raw
  // pointers plus the shape, layout and dtype to read them with. The mixed
  // ops fill the last four; the single-format ops leave them null, which is
  // what selects the kernel's MIXED=false instantiation at the call site.
  //
  // One batch dimension (X1). `batch` is the broadcast batch size and the two
  // operand strides are *element* offsets between consecutive batch elements,
  // where **0 means broadcast** -- a `[K, N]` weight shared by every batch
  // element rides at stride 0 rather than being materialized B times. C is
  // always written densely as [batch, M, N], so its stride is M*N and is not
  // carried. A 2D call is batch = 1 with both strides 0, which is the same
  // arithmetic every element ran before: nothing downstream branches on it.
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

  // Shared across every palette entry of a mixed op: only the format *widths*
  // are tabulated per slot, matching the schemas, which take the sign,
  // saturation, subnormal and prng-bit settings as scalars.
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

  // One palette slot's widths. `n` slots of these ride alongside a *Common.
  struct BinaryKWidths
  {
    int man_bits = 0, exp_bits = 0, bias = 0;
  };

  struct SuperfpWidths
  {
    int man_bits = 0, exp_bits = 0, normal_binades = 0, bias = 0;
  };

  // The schemas spell a binaryK format as (K, P) -- total bits and precision,
  // the two parameters IEEE P3109 names its binaryK formats by -- where the
  // policies want (man_bits, exp_bits). One conversion, here, rather than the
  // same two lines in each of the four entry points that takes that spelling.
  inline BinaryKWidths binaryK_widths(int64_t K, int64_t P, int64_t bias, bool is_signed)
  {
    return BinaryKWidths{static_cast<int>(P - 1),
                         static_cast<int>(is_signed ? K - P : K - P + 1),
                         static_cast<int>(bias)};
  }

  inline SuperfpWidths superfp_widths(int64_t man_bits, int64_t exp_bits, int64_t normal_binades,
                                      int64_t bias)
  {
    return SuperfpWidths{static_cast<int>(man_bits), static_cast<int>(exp_bits),
                         static_cast<int>(normal_binades), static_cast<int>(bias)};
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE BinaryKMultiplierT<T, RM> make_mul(const BinaryKWidths &w, const BinaryKCommon &c)
  {
    return BinaryKMultiplierT<T, RM>{w.man_bits, w.exp_bits, w.bias, c.is_signed, c.sat, c.sub, c.prng_bits};
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE BinaryKAdderT<T, RM> make_add(const BinaryKWidths &w, const BinaryKCommon &c)
  {
    return BinaryKAdderT<T, RM>{w.man_bits, w.exp_bits, w.bias, c.is_signed, c.sat, c.sub, c.prng_bits};
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE SuperfpMultiplierT<T, RM> make_mul(const SuperfpWidths &w, const SuperfpCommon &c)
  {
    return SuperfpMultiplierT<T, RM>{w.man_bits, w.exp_bits, w.normal_binades, w.bias, c.is_signed, c.sat,
                                  c.prng_bits};
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE SuperfpAdderT<T, RM> make_add(const SuperfpWidths &w, const SuperfpCommon &c)
  {
    return SuperfpAdderT<T, RM>{w.man_bits, w.exp_bits, w.normal_binades, w.bias, c.is_signed, c.sat,
                             c.prng_bits};
  }

  // ----------------------------------------------------------------------
  // binaryK, split mac (custom_matmul_binaryK)
  // ----------------------------------------------------------------------
  struct BinaryKSplitArgs
  {
    static constexpr bool mixed = false;
    // SplitMac draws for the multiply and the accumulate independently, so a
    // thread's K-step reduction consumes two Philox values per step where a
    // FusedMac consumes one. Only RoundMode::SR draws at all; this is the
    // upper bound matmul_rng_engine_inputs reserves against. A draw is one
    // word in binary32 and two in binary64, which the driver multiplies in.
    static constexpr uint64_t draws_per_k_step = 2;

    BinaryKWidths mul{};
    BinaryKCommon mul_c{};
    bool accumulate_quant = false;
    BinaryKWidths acc{};
    BinaryKCommon acc_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(F &&f) const
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
    static constexpr bool mixed = true;
    static constexpr uint64_t draws_per_k_step = 2;

    int n_fmt = 0;
    BinaryKWidths mul[MAX_GEMM_FORMATS]{};
    BinaryKCommon mul_c{};
    bool accumulate_quant = false;
    BinaryKWidths acc[MAX_GEMM_FORMATS]{};
    BinaryKCommon acc_c{};

    // This Mac was the biggest of the eight at 112 registers, and finding
    // G10's derived-params treatment -- rebuilding BinaryKParams' six
    // fast-path floats where they are read instead of carrying them -- only
    // took it to 99, where a third resident block needed 80. It was built,
    // measured at 0.86x, and never taken; K1 has since made the question moot
    // by taking this kernel to 64-80 registers on its own. Measured, not
    // assumed; see dev/gemm_perf_audit.md (G10) and the roadmap (H3).
    template <class T, RoundMode RM, class F>
    void with_palette(F &&f) const
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
    static constexpr bool mixed = false;
    static constexpr uint64_t draws_per_k_step = 1;

    bool fma_quant = false;
    BinaryKWidths fma{};
    BinaryKCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(F &&f) const
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
    static constexpr bool mixed = true;
    static constexpr uint64_t draws_per_k_step = 1;

    int n_fmt = 0;
    BinaryKWidths fma[MAX_GEMM_FORMATS]{};
    BinaryKCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_palette(F &&f) const
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
    static constexpr bool mixed = false;
    static constexpr uint64_t draws_per_k_step = 2;

    SuperfpWidths mul{};
    SuperfpCommon mul_c{};
    bool accumulate_quant = false;
    SuperfpWidths acc{};
    SuperfpCommon acc_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(F &&f) const
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
    static constexpr bool mixed = true;
    static constexpr uint64_t draws_per_k_step = 2;

    int n_fmt = 0;
    SuperfpWidths mul[MAX_GEMM_FORMATS]{};
    SuperfpCommon mul_c{};
    bool accumulate_quant = false;
    SuperfpWidths acc[MAX_GEMM_FORMATS]{};
    SuperfpCommon acc_c{};

    // Two full superfp policies per palette slot was the one Mac of the eight
    // that ran out of registers on the GPU -- 100 registers, 2 blocks resident
    // per SM where its single-format twin got 5 -- which is what finding G10's
    // second SuperfpParams spelling existed to buy back. K1 took this kernel
    // to 60-64 registers on its own, so there is nothing left to buy and the
    // spelling is gone; see dev/gemm_roadmap.md (finding H3).
    template <class T, RoundMode RM, class F>
    void with_palette(F &&f) const
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
    static constexpr bool mixed = false;
    static constexpr uint64_t draws_per_k_step = 1;

    bool fma_quant = false;
    SuperfpWidths fma{};
    SuperfpCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(F &&f) const
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

  struct SuperfpFusedMixedArgs
  {
    static constexpr bool mixed = true;
    static constexpr uint64_t draws_per_k_step = 1;

    int n_fmt = 0;
    SuperfpWidths fma[MAX_GEMM_FORMATS]{};
    SuperfpCommon fma_c{};

    template <class T, RoundMode RM, class F>
    void with_palette(F &&f) const
    {
      using Mac = FusedMac<SuperfpAdderT<T, RM>>;
      FormatPalette<Mac> pal;
      for (int i = 0; i < n_fmt; ++i)
        pal.slots[i] = Mac{make_add<T, RM>(fma[i], fma_c)};
      f(NaiveAccumulator<Mac>{pal.slots[0], T(0)}, pal);
    }
  };

} // namespace mptorch::gemm
