#pragma once

// The host side of the counted GEMMs (gemm_counted.h): the drivers of the
// counted GEMM and conv ops, their checks, and the refusal of a build
// without MPTORCH_COUNTING=1.
//
// A counted op takes its *_accumulated twin's schema and every algorithm,
// NAIVE included, and returns the result with an int64 [ROLES, EVENTS]
// tensor of counts (event_counts.h's Role and Event). The driver allocates
// the counts, then runs the shared driver, run_custom_matmul (gemm_host.h)
// or run_custom_conv (gemm_conv_host.h), on a CountingArgs of the call's
// algorithm, so the checks, their order and their messages are the twins'.
// A conv op's input gradient runs one GEMM per residue class of the stride,
// and all of them add to the one tensor of counts. This header defines the overloads of
// run_custom_matmul's hooks for a CountingArgs, which argument-dependent
// lookup finds where the driver is instantiated.

#include "gemm_conv_host.h"
#include "gemm_counted.h"
#include "gemm_host.h"

#include <ATen/ops/zeros.h>
#include <tuple>

namespace mptorch::gemm
{
#if !defined(MPTORCH_COUNTING)
  // A build without MPTORCH_COUNTING=1 (setup.py) has no counted kernels.
  // The counted ops raise here, naming the flag, from their entry points and
  // Meta kernels, before anything instantiates a driver on a CountingArgs,
  // so nothing refers to the units the flag leaves out.
  [[noreturn]] inline void counting_unavailable(const char *op_name)
  {
    TORCH_CHECK(false, op_name,
                ": this build has no counted GEMM kernels (they are built with "
                "MPTORCH_COUNTING=1); rebuild with the flag to count numeric events");
  }
#endif

  // A NAIVE call takes no block size and no outer format; the others are
  // checked as their *_accumulated twins check them.
  template <class Base, AccumulateAlgorithm ALG>
  void check_accumulate_algorithm(const CountingArgs<Base, ALG> &args, const char *op_name,
                                  int64_t accumulate_algorithm)
  {
    if constexpr (ALG == AccumulateAlgorithm::NAIVE)
    {
      TORCH_CHECK(args.block_size == 0 && !args.outer_quant, op_name,
                  ": AccumulateAlgorithm.NAIVE takes no block_size and no outer format");
    }
    else
    {
      AccumulateArgs<Base> acc;
      acc.base = args.base;
      acc.alg = ALG;
      acc.block_size = args.block_size;
      acc.outer_quant = args.outer_quant;
      acc.outer = args.outer;
      acc.outer_c = args.outer_c;
      check_accumulate_algorithm(acc, op_name, accumulate_algorithm);
    }
  }

  template <class Base, AccumulateAlgorithm ALG>
  uint64_t draws_per_k_step_of(const CountingArgs<Base, ALG> &)
  {
    return accumulate_draws_per_k_step(ALG, Base::draws_per_k_step);
  }

  template <AccumulateAlgorithm ALG, class Base, class Widths, class Common>
  CountingArgs<Base, ALG> counting_args(const Base &base, const AccumulateTail<Widths, Common> &tail,
                                        unsigned long long *counts)
  {
    CountingArgs<Base, ALG> args;
    args.base = base;
    args.block_size = static_cast<int>(tail.block_size);
    args.outer_quant = tail.outer_quant;
    args.outer = tail.outer;
    args.outer_c = tail.outer_c;
    args.counts = counts;
    return args;
  }

  // The counted twin of run_custom_matmul_accumulated: the same checks of
  // the algorithm and the block size, then the shared driver on the
  // algorithm's CountingArgs.
  template <class Backend, class Base, class Widths, class Common>
  std::tuple<Tensor, Tensor> run_custom_matmul_counted(const char *op_name, const Base &base,
                                                       const AccumulateTail<Widths, Common> &tail,
                                                       Tensor a, Tensor b, bool trans_a, bool trans_b,
                                                       int64_t accumulate_algorithm, int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    counting_unavailable(op_name);
#else
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    auto *p = reinterpret_cast<unsigned long long *>(counts.data_ptr<int64_t>());
    Tensor c;
    switch (static_cast<AccumulateAlgorithm>(accumulate_algorithm))
    {
    case AccumulateAlgorithm::NAIVE:
      c = run_custom_matmul<Backend>(op_name, counting_args<AccumulateAlgorithm::NAIVE>(base, tail, p),
                                     a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    case AccumulateAlgorithm::KAHAN:
      c = run_custom_matmul<Backend>(op_name, counting_args<AccumulateAlgorithm::KAHAN>(base, tail, p),
                                     a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    case AccumulateAlgorithm::BLOCK:
      c = run_custom_matmul<Backend>(op_name, counting_args<AccumulateAlgorithm::BLOCK>(base, tail, p),
                                     a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    case AccumulateAlgorithm::TREE:
      // over a fused mac, check_accumulate_algorithm refuses TREE before the
      // launch, which then builds no kernel
      c = run_custom_matmul<Backend>(op_name, counting_args<AccumulateAlgorithm::TREE>(base, tail, p),
                                     a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
      break;
    }
    return {c, counts};
#endif
  }

  // The counted twin of run_custom_conv_single: its checks of the algorithm
  // and the accumulation arguments, in its order, then the conv driver on the
  // algorithm's CountingArgs.
  template <class Backend, class Base, class Widths, class Common>
  std::tuple<Tensor, Tensor> run_custom_conv_counted(const char *op_name, const Base &base,
                                                     const AccumulateTail<Widths, Common> &tail,
                                                     Tensor a, Tensor b, int64_t conv_pass,
                                                     c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                                     c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                                     int64_t groups, int64_t accumulate_algorithm,
                                                     int64_t round_mode)
  {
#if !defined(MPTORCH_COUNTING)
    counting_unavailable(op_name);
#else
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    Tensor counts = at::zeros({events::ROLES, events::EVENTS}, a.options().dtype(at::kLong));
    auto *p = reinterpret_cast<unsigned long long *>(counts.data_ptr<int64_t>());
    const auto run = [&](const auto &args)
    {
      check_accumulate_algorithm(args, op_name, accumulate_algorithm);
      return run_custom_conv<Backend>(op_name, args, a, b, nullptr, conv_pass, out_size, stride,
                                      padding, dilation, groups, round_mode);
    };
    Tensor c;
    switch (static_cast<AccumulateAlgorithm>(accumulate_algorithm))
    {
    case AccumulateAlgorithm::NAIVE:
      c = run(counting_args<AccumulateAlgorithm::NAIVE>(base, tail, p));
      break;
    case AccumulateAlgorithm::KAHAN:
      c = run(counting_args<AccumulateAlgorithm::KAHAN>(base, tail, p));
      break;
    case AccumulateAlgorithm::BLOCK:
      c = run(counting_args<AccumulateAlgorithm::BLOCK>(base, tail, p));
      break;
    case AccumulateAlgorithm::TREE:
      c = run(counting_args<AccumulateAlgorithm::TREE>(base, tail, p));
      break;
    }
    return {c, counts};
#endif
  }
} // namespace mptorch::gemm
