#pragma once

// The Meta twins of the counted GEMM and conv drivers (gemm_counted_host.h,
// fixedpoint_counted_host.h).
//
// gemm_counted_host.h comes first: meta_custom_matmul calls
// g::check_accumulate_algorithm by its qualified name, which sees only the
// overloads declared before meta_host.h defines it, and a CountingArgs needs
// its own.

#include "fixedpoint_counted_host.h"
#include "gemm_counted_host.h"
#include "meta_host.h"

#include "fixedpoint_meta_host.h"

#include <ATen/ops/zeros.h>
#include <tuple>

namespace mptorch::meta
{
  template <class Base, class Widths, class Common>
  std::tuple<Tensor, Tensor> meta_custom_matmul_counted(const char *op_name, const Base &base,
                                                        const g::AccumulateTail<Widths, Common> &tail,
                                                        const Tensor &a, const Tensor &b, bool trans_a,
                                                        bool trans_b, int64_t accumulate_algorithm,
                                                        int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    g::counting_unavailable(op_name);
#else
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    Tensor c;
    switch (static_cast<AccumulateAlgorithm>(accumulate_algorithm))
    {
    case AccumulateAlgorithm::NAIVE:
      c = meta_custom_matmul(op_name, g::counting_args<AccumulateAlgorithm::NAIVE>(base, tail, nullptr),
                             a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    case AccumulateAlgorithm::KAHAN:
      c = meta_custom_matmul(op_name, g::counting_args<AccumulateAlgorithm::KAHAN>(base, tail, nullptr),
                             a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    case AccumulateAlgorithm::BLOCK:
      c = meta_custom_matmul(op_name, g::counting_args<AccumulateAlgorithm::BLOCK>(base, tail, nullptr),
                             a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    case AccumulateAlgorithm::TREE:
      c = meta_custom_matmul(op_name, g::counting_args<AccumulateAlgorithm::TREE>(base, tail, nullptr),
                             a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    }
    return {c, counts};
#endif
  }

  // The twin of run_custom_conv_counted (gemm_counted_host.h).
  template <class Base, class Widths, class Common>
  std::tuple<Tensor, Tensor> meta_custom_conv_counted(const char *op_name, const Base &base,
                                                      const g::AccumulateTail<Widths, Common> &tail,
                                                      const Tensor &a, const Tensor &b,
                                                      int64_t conv_pass, c10::IntArrayRef out_size,
                                                      c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                      c10::IntArrayRef dilation, int64_t groups,
                                                      int64_t accumulate_algorithm, int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    g::counting_unavailable(op_name);
#else
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    const auto run = [&](const auto &args)
    {
      g::check_accumulate_algorithm(args, op_name, accumulate_algorithm);
      return meta_custom_conv(op_name, args, a, b, nullptr, conv_pass, out_size, stride, padding,
                              dilation, groups, round_mode);
    };
    Tensor c;
    switch (static_cast<AccumulateAlgorithm>(accumulate_algorithm))
    {
    case AccumulateAlgorithm::NAIVE:
      c = run(g::counting_args<AccumulateAlgorithm::NAIVE>(base, tail, nullptr));
      break;
    case AccumulateAlgorithm::KAHAN:
      c = run(g::counting_args<AccumulateAlgorithm::KAHAN>(base, tail, nullptr));
      break;
    case AccumulateAlgorithm::BLOCK:
      c = run(g::counting_args<AccumulateAlgorithm::BLOCK>(base, tail, nullptr));
      break;
    case AccumulateAlgorithm::TREE:
      c = run(g::counting_args<AccumulateAlgorithm::TREE>(base, tail, nullptr));
      break;
    }
    return {c, counts};
#endif
  }

  // The twins of run_custom_{matmul,conv}_fixedpoint_counted
  // (fixedpoint_counted_host.h): the counts, then the fixed-point Meta
  // kernel's checks and result.
  inline std::tuple<Tensor, Tensor> meta_custom_matmul_fixedpoint_counted(
      const char *op_name, const g::FixedPointGemmCall &call, const Tensor &a, const Tensor &b,
      bool trans_a, bool trans_b, int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    g::counting_unavailable(op_name);
#else
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    Tensor c = meta_custom_matmul_fixedpoint(op_name, call, a, b, trans_a, trans_b, round_mode);
    return {c, counts};
#endif
  }

  inline std::tuple<Tensor, Tensor> meta_custom_conv_fixedpoint_counted(
      const char *op_name, const g::FixedPointGemmCall &call, const Tensor &a, const Tensor &b,
      int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride,
      c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    g::counting_unavailable(op_name);
#else
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    Tensor c = meta_custom_conv_fixedpoint(op_name, call, a, b, conv_pass, out_size, stride,
                                           padding, dilation, groups, round_mode);
    return {c, counts};
#endif
  }
} // namespace mptorch::meta

