#pragma once

// The cross-family split mac (custom_matmul_binaryK_superfp and
// custom_matmul_superfp_binaryK, their *_accumulated twins and their conv
// ops): each product rounded to a format of one family and each partial sum
// to a format of the other. The policies always composed this way, since
// make_mul and make_add overload on the widths struct (common/gemm_args.h);
// what is new is an Args that names two families, and an AccumulateFamily
// saying that BLOCK's and TREE's outer format is of the accumulate's.
//
// A header of its own, which gemm_args.h and gemm_policy.h do not include, so
// that the kernels every other op runs compile to the bytes they compiled
// before (dev/gemm_roadmap.md, R-1). The Metal prelude does not include it:
// the MPS backend binds these ops to functions that raise until
// dev/continuation_plan.md's phase H. It is written the way the other common/
// headers are, MPTORCH_THREAD on references and no lambdas, so that phase H
// ports it rather than rewrites it.
//
// The accumulate format is always there. A split mac whose sum stays in the
// carrier has one family only, and is that family's op with
// accumulate_quant=false; so there is no flag here, and no IdentityAdder
// instantiation.

#include "gemm_accumulate.h"
#include "gemm_args.h"

namespace mptorch::gemm
{

  template <class MulW, class MulC, class AccW, class AccC>
  struct CrossSplitArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    // As a single-family split mac: one draw for the product and one for the
    // sum per K-step under RoundMode::SR.
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

  // binaryK products summed in a superfp format, and the reverse.
  using BinaryKSuperfpArgs = CrossSplitArgs<BinaryKWidths, BinaryKCommon, SuperfpWidths, SuperfpCommon>;
  using SuperfpBinaryKArgs = CrossSplitArgs<SuperfpWidths, SuperfpCommon, BinaryKWidths, BinaryKCommon>;

  // KAHAN, BLOCK and TREE over a cross-family mac. The outer format is of the
  // accumulate's family: it rounds a total of sums that are already the
  // accumulate format's values.
  template <class MulW, class MulC, class AccW, class AccC>
  struct AccumulateFamily<CrossSplitArgs<MulW, MulC, AccW, AccC>>
  {
    using Widths = AccW;
    using Common = AccC;
    static constexpr MPTORCH_CONSTANT bool has_product = true;
  };

} // namespace mptorch::gemm
