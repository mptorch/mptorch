#pragma once

// The Meta twins of run_custom_matmul_fixedpoint (fixedpoint_gemm_host.h) and
// run_custom_conv_fixedpoint (fixedpoint_conv_host.h), for
// meta/custom_{matmul,conv}_fixedpoint_entry.cpp: the CPU driver's checks in
// its order, with its messages, and its allocation on symbolic sizes,
// without a data pointer, a launch or a draw (meta_host.h says why). In a
// header of its own, so that meta_host.h, which every other Meta kernel
// includes, is unchanged.

#include "fixedpoint_gemm_host.h"
#include "meta_host.h"

namespace mptorch::meta
{
  inline Tensor meta_custom_matmul_fixedpoint(const char *op_name,
                                              const g::FixedPointGemmCall &call_args,
                                              const Tensor &a, const Tensor &b, bool trans_a,
                                              bool trans_b, int64_t round_mode)
  {
    g::check_matmul_inputs(a, b, op_name, round_mode,
                           static_cast<int64_t>(AccumulateAlgorithm::NAIVE));
    g::FixedPointGemmCall call = call_args;
    const g::FixedPointGemmPlan plan =
        g::plan_fixedpoint_gemm(call, round_mode, a.scalar_type(), op_name);
    const SymMatmulShape s = sym_matmul_output_shape(a, b, trans_a, trans_b, op_name);
    Tensor c = sym_empty(sym_output_sizes(s.batch, s.M, s.N, s.batched),
                         a.options().dtype(plan.wide ? at::kDouble : a.scalar_type()));
    if (sym_is_empty(s.batch, s.M, s.N))
      return c;
    mptorch::gemm_dtype_of(a, b, op_name);
    return c;
  }

  inline Tensor meta_custom_conv_fixedpoint(const char *op_name,
                                            const g::FixedPointGemmCall &call_args, const Tensor &a,
                                            const Tensor &b, int64_t conv_pass,
                                            c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                            c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                            int64_t groups, int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op_name, ": ", round_mode,
                " is not a RoundMode");
    g::FixedPointGemmCall call = call_args;
    const g::FixedPointGemmPlan plan =
        g::plan_fixedpoint_gemm(call, round_mode, a.scalar_type(), op_name);
    const SymConvShape cs =
        sym_conv_shape(a, b, conv_pass, out_size, stride, padding, dilation, groups, op_name);
    Tensor c =
        sym_empty(cs.out_sizes, a.options().dtype(plan.wide ? at::kDouble : a.scalar_type()));
    if (sym_is_empty(cs.batch, cs.M, cs.N))
      return c;
    mptorch::gemm_dtype_of(a, b, op_name);
    return c;
  }
} // namespace mptorch::meta
