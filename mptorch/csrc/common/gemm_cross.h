#pragma once

// The cross-family split mac (custom_matmul_binaryK_superfp and
// custom_matmul_superfp_binaryK, their *_accumulated twins and their conv
// ops): each product is rounded to a format of one family, and each partial
// sum to a format of the other. The policies could always be combined this
// way, since make_mul and make_add are overloaded on the widths struct
// (gemm_args.h). What this header adds is an Args that names two families,
// and an AccumulateFamily that gives BLOCK's and TREE's outer format the
// accumulate format's family.
//
// gemm_args.h and gemm_policy.h do not include this header, so that the
// kernels of every other op compile exactly as they would without it. The
// Metal prelude includes it, so it is written like the other common/
// headers, with MPTORCH_THREAD on references and no lambdas.
//
// The accumulate format is always present. A split mac whose sum stays in
// the carrier has only one family, and is that family's op with
// accumulate_quant=false; so there is no such flag here, and no IdentityAdder
// instantiation.

#include "gemm_accumulate.h"
#include "gemm_args.h"

namespace mptorch::gemm
{

  template <class MulW, class MulC, class AccW, class AccC>
  struct CrossSplitArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    // As for a single-family split mac: one draw for the product and one for
    // the sum per K-step under RoundMode::SR.
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 2;

    MulW mul{};
    MulC mul_c{};
    AccW acc{};
    AccC acc_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      using Mac = SplitMac<decltype(make_mul<T, RM>(mul, mul_c)), decltype(make_add<T, RM>(acc, acc_c))>;
      f(NaiveAccumulator<Mac>{Mac{make_mul<T, RM>(mul, mul_c), make_add<T, RM>(acc, acc_c)}, T(0)});
    }
  };

  // binaryK products summed in a superfp format, and superfp products summed
  // in a binaryK format.
  using BinaryKSuperfpArgs = CrossSplitArgs<BinaryKWidths, BinaryKCommon, SuperfpWidths, SuperfpCommon>;
  using SuperfpBinaryKArgs = CrossSplitArgs<SuperfpWidths, SuperfpCommon, BinaryKWidths, BinaryKCommon>;

  // KAHAN, BLOCK and TREE over a cross-family mac. The outer format is in the
  // accumulate format's family, because it rounds a total of sums that are
  // already values of the accumulate format.
  template <class MulW, class MulC, class AccW, class AccC>
  struct AccumulateFamily<CrossSplitArgs<MulW, MulC, AccW, AccC>>
  {
    using Widths = AccW;
    using Common = AccC;
    static constexpr MPTORCH_CONSTANT bool has_product = true;
  };

} // namespace mptorch::gemm
