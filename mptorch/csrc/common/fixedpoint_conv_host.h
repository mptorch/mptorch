#pragma once

// The host side of the fixed-point conv ops: run_custom_conv's steps
// (gemm_conv_host.h), in its order, over the fixed-point GEMM's checks, plan
// and Args (fixedpoint_gemm_host.h). A conv op's schema is the GEMM op's
// with the geometry in front of the formats, as for the other conv ops:
//
//   custom_conv_fixedpoint[_fma](Tensor a, Tensor b, int conv_pass,
//       int[] out_size, int[] stride, int[] padding, int[] dilation,
//       int groups, <the GEMM op's format arguments>)
//
// NAIVE only, as the fixed-point GEMM is. Each pass is the fixed-point GEMM
// over the gathered operands, and the input gradient is one GEMM per residue
// class of the stride (gemm_conv_host.h's conv_igrad_classes), whose skipped
// terms are zeros: a zero product leaves a fixed-point sum as it was, so the
// pass equals the zero-inserted sum in every mode but stochastic rounding,
// where the zero-inserted sum would draw for them. The result is float64 when
// the plan says it is wide.
//
// A driver of its own, like run_custom_matmul_fixedpoint, so that
// gemm_conv_host.h, which the other conv ops' entry files include, is
// unchanged.

#include "fixedpoint_gemm_host.h"
#include "gemm_conv_host.h"

namespace mptorch::gemm
{
  // Launches one pass on `inner`, the plan's Args: once for the forward and
  // the weight gradient, once per residue class for the input gradient, each
  // with its own random-state draw, as run_custom_conv does.
  template <class Backend, class Inner>
  void launch_conv_fixedpoint_as(GemmShape &s, const ConvShape &cs, const Inner &inner,
                                 const Tensor &a, const Tensor &b, const Tensor &c)
  {
    ConvArgs<Inner> args;
    args.inner = inner;
    const uint64_t draws = draws_per_k_step_of(inner) * words_per_draw(s.dt);
    if (cs.geom.pass != ConvPass::IGRAD)
    {
      typename Backend::LaunchContext ctx =
          Backend::make_context(s.use_rng, draws * static_cast<uint64_t>(s.K));
      bind_tensors(ctx, a, b, c, nullptr);
      args.geom = cs.geom;
      Backend::launch(s, args, ctx);
      return;
    }
    for (const ConvClass &cl : conv_igrad_classes(cs.geom))
    {
      s.K = cl.K;
      s.N = cl.N;
      typename Backend::LaunchContext ctx =
          Backend::make_context(s.use_rng, draws * static_cast<uint64_t>(s.K));
      bind_tensors(ctx, a, b, c, nullptr);
      args.geom = cl.geom;
      Backend::launch(s, args, ctx);
    }
  }

  template <class Backend>
  void launch_conv_fixedpoint(GemmShape &s, const ConvShape &cs, const FixedPointGemmCall &call,
                              const FixedPointGemmPlan &plan, const Tensor &a, const Tensor &b,
                              const Tensor &c)
  {
    if (call.fused)
    {
      if (call.wrap)
        launch_conv_fixedpoint_as<Backend>(
            s, cs, fixedpoint_args<FixedPointIntFusedArgs<true>>(call, plan), a, b, c);
      else
        launch_conv_fixedpoint_as<Backend>(
            s, cs, fixedpoint_args<FixedPointIntFusedArgs<false>>(call, plan), a, b, c);
    }
    else if (plan.carrier_path)
      launch_conv_fixedpoint_as<Backend>(s, cs, fixedpoint_args<FixedPointSplitArgs>(call, plan), a,
                                         b, c);
    else if (call.wrap)
      launch_conv_fixedpoint_as<Backend>(
          s, cs, fixedpoint_args<FixedPointIntSplitArgs<true>>(call, plan), a, b, c);
    else
      launch_conv_fixedpoint_as<Backend>(
          s, cs, fixedpoint_args<FixedPointIntSplitArgs<false>>(call, plan), a, b, c);
  }

  // run_custom_conv for a fixed-point call, step for step, with the result's
  // dtype from the plan.
  template <class Backend>
  Tensor run_custom_conv_fixedpoint(const char *op_name, const FixedPointGemmCall &call_args,
                                    Tensor a, Tensor b, int64_t conv_pass,
                                    c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                    c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                    int64_t groups, int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op_name, ": ", round_mode,
                " is not a RoundMode");
    FixedPointGemmCall call = call_args;
    const FixedPointGemmPlan plan =
        plan_fixedpoint_gemm(call, round_mode, a.scalar_type(), op_name);
    // MPS has no float64 tensors (run_custom_matmul_fixedpoint's check).
    TORCH_CHECK(!(plan.wide && a.is_mps()), op_name, ": the accumulate format has ",
                call.acc.mag(), " magnitude bits, more than ",
                a.scalar_type() == at::kHalf       ? "float16"
                : a.scalar_type() == at::kBFloat16 ? "bfloat16"
                                                   : "float32",
                "'s ", fixedpoint_storage_precision(a.scalar_type()),
                ", so the result is float64, which MPS does not have: use an accumulate format "
                "of at most ",
                fixedpoint_storage_precision(a.scalar_type()),
                " magnitude bits, or compute on the CPU");
    const ConvShape cs =
        conv_shape(a, b, conv_pass, out_size, stride, padding, dilation, groups, op_name);

    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    Tensor c =
        at::empty(cs.out_sizes, a.options().dtype(plan.wide ? at::kDouble : a.scalar_type()));
    if (cs.batch == 0 || cs.M == 0 || cs.N == 0)
      return c;

    GemmShape s;
    s.M = cs.M;
    s.K = cs.K;
    s.N = cs.N;
    s.batch = cs.batch;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);
    s.dt = mptorch::gemm_dtype_of(a_c, b_c, op_name);
    s.a = a_c.data_ptr();
    s.b = b_c.data_ptr();
    s.c = c.data_ptr();

    launch_conv_fixedpoint<Backend>(s, cs, call, plan, a_c, b_c, c);
    return c;
  }
} // namespace mptorch::gemm
