#pragma once

// The host side of the counted fixed-point GEMM and conv ops
// (fixedpoint_counted.h).
//
// A fixed-point call is planned (fixedpoint_gemm_host.h's
// plan_fixedpoint_gemm: the carrier path or the integer accumulator, split
// or fused, saturating or wrapping, wide or not) and then launched by its
// driver. The counted drivers run those same drivers, through
// CountingBackend, a backend whose launch wraps the plan's Args in a
// FixedPointCountingArgs (inside its ConvArgs for a conv pass), so a counted
// call is planned, checked and launched as its twin is. The counts reach the
// launch through fixedpoint_counts, which the counted driver sets on its own
// thread for the length of the call: the shared driver launches on that
// thread, and takes no argument for them.

#include "fixedpoint_conv_host.h"
#include "fixedpoint_counted.h"
#include "fixedpoint_gemm_host.h"
#include "gemm_counted_host.h"

#include <ATen/ops/zeros.h>
#include <tuple>

namespace mptorch::gemm
{
  // Where the current counted fixed-point call's counts go.
  inline thread_local unsigned long long *fixedpoint_counts = nullptr;

  // Sets fixedpoint_counts for the length of one call.
  struct FixedPointCountsScope
  {
    explicit FixedPointCountsScope(unsigned long long *counts) { fixedpoint_counts = counts; }
    ~FixedPointCountsScope() { fixedpoint_counts = nullptr; }
    FixedPointCountsScope(const FixedPointCountsScope &) = delete;
    FixedPointCountsScope &operator=(const FixedPointCountsScope &) = delete;
  };

  template <class Backend>
  struct CountingBackend
  {
    using LaunchContext = typename Backend::LaunchContext;

    static LaunchContext make_context(bool use_rng, uint64_t draws_per_thread)
    {
      return Backend::make_context(use_rng, draws_per_thread);
    }

    template <class Args>
    static void launch(const GemmShape &s, const Args &args, const LaunchContext &ctx)
    {
      Backend::launch(s, FixedPointCountingArgs<Args>{args, fixedpoint_counts}, ctx);
    }

    template <class Inner>
    static void launch(const GemmShape &s, const ConvArgs<Inner> &args, const LaunchContext &ctx)
    {
      ConvArgs<FixedPointCountingArgs<Inner>> counted;
      counted.inner = FixedPointCountingArgs<Inner>{args.inner, fixedpoint_counts};
      counted.geom = args.geom;
      Backend::launch(s, counted, ctx);
    }
  };

  template <class Backend>
  std::tuple<Tensor, Tensor> run_custom_matmul_fixedpoint_counted(const char *op_name,
                                                                  const FixedPointGemmCall &call,
                                                                  Tensor a, Tensor b, bool trans_a,
                                                                  bool trans_b, int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    counting_unavailable(op_name);
#else
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    const FixedPointCountsScope scope(reinterpret_cast<unsigned long long *>(counts.data_ptr<int64_t>()));
    Tensor c = run_custom_matmul_fixedpoint<CountingBackend<Backend>>(op_name, call, a, b, trans_a,
                                                                      trans_b, round_mode);
    return {c, counts};
#endif
  }

  template <class Backend>
  std::tuple<Tensor, Tensor> run_custom_conv_fixedpoint_counted(
      const char *op_name, const FixedPointGemmCall &call, Tensor a, Tensor b, int64_t conv_pass,
      c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding,
      c10::IntArrayRef dilation, int64_t groups, int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    counting_unavailable(op_name);
#else
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    const FixedPointCountsScope scope(reinterpret_cast<unsigned long long *>(counts.data_ptr<int64_t>()));
    Tensor c = run_custom_conv_fixedpoint<CountingBackend<Backend>>(
        op_name, call, a, b, conv_pass, out_size, stride, padding, dilation, groups, round_mode);
    return {c, counts};
#endif
  }
} // namespace mptorch::gemm
