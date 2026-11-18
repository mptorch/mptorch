#pragma once

// The MPS backend as common/gemm_host.h's driver sees it: a launch context
// and one `launch` per kind of Args (the struct that carries an op's formats
// and builds its policy), with the shape of cpu/gemm_backend.h and
// cuda/gemm_backend.h. What differs is where the policies are built. The CPU
// and CUDA backends build them on the host and hand the kernel the result;
// this one hands the kernel the Args themselves, written out as Metal source
// (args_source below), and the kernel builds the policies from them with
// the Args' own factories: with_accumulator, as the other backends do, and
// on the mixed ops with_slot, one element's slot at a time, where the other
// backends take with_palette's whole palette (gemm.metal says why). That
// makes every format constant a compile-time constant of the kernel, at the
// price of one compile per distinct format (metal_runtime.h).
//
// The ops beyond the eight NAIVE GEMMs run the same kernel with a variant
// named in the instantiation (gemm.metal): an AccumulateArgs adds the
// algorithm (KAHAN, BLOCK or TREE), a ConvArgs the gathered loads and the
// pass, with the geometry bound as a buffer, and a BlockGemmArgs the decode
// of packed operands, with their layouts and scales bound as buffers. The
// cross-family Args need no variant.
//
// binary32 only: MPS has no float64 tensors, so there is no binary64 kernel
// to pick and nothing for MPTORCH_NO_FP64 to leave out.
//
// The Args of the odd carrier step (OddArgs, OddPaletteArgs, and an
// AccumulateArgs over an OddArgs) run the same kernel, compiled after
// mps/odd.metal, which holds the headers that define them; uses_odd_kernel
// decides that, and nothing else. Their args_source are templates,
// instantiated in gemm_backend.cpp for the sixteen Args the odd ops build.

#include "../common/gemm_accumulate.h"
#include "../common/gemm_args.h"
#include "../common/gemm_block.h"
#include "../common/gemm_cross.h"
#include "../common/gemm_gather.h"
#include <ATen/core/Tensor.h>
#include <cstdint>
#include <string>

namespace mptorch::gemm
{
  template <class Base>
  struct OddArgs;
  template <class Base>
  struct OddPaletteArgs;
} // namespace mptorch::gemm

namespace mptorch::gemm_mps
{

  // Whether an Args belongs to the odd carrier step, whose kernel is
  // compiled after mps/odd.metal (KernelSource::GemmOdd).
  template <class Args>
  struct uses_odd_kernel
  {
    static constexpr bool value = false;
  };
  template <class Base>
  struct uses_odd_kernel<mptorch::gemm::OddArgs<Base>>
  {
    static constexpr bool value = true;
  };
  template <class Base>
  struct uses_odd_kernel<mptorch::gemm::OddPaletteArgs<Base>>
  {
    static constexpr bool value = true;
  };
  template <class Base>
  struct uses_odd_kernel<mptorch::gemm::AccumulateArgs<Base>> : uses_odd_kernel<Base>
  {
  };

  struct MpsBackend
  {
    // The SR seed, drawn as the CPU backend draws it, and the tensors, which
    // the driver binds (gemm_host.h's bind_tensors, and the block GEMM's
    // scales through gemm_block_host.h's bind_block_scales), because a
    // kernel takes an MPS tensor as its MTLBuffer and a byte offset rather
    // than as the pointer GemmShape carries. The tensors are the driver's
    // locals and outlive the launch.
    struct LaunchContext
    {
      uint64_t seed = 0;
      const at::Tensor *a = nullptr;
      const at::Tensor *b = nullptr;
      const at::Tensor *c = nullptr;
      const at::Tensor *prec_idx = nullptr;
      const at::Tensor *scales_a = nullptr;
      const at::Tensor *scales_b = nullptr;

      void bind(const at::Tensor &a_, const at::Tensor &b_, const at::Tensor &c_,
                const at::Tensor *prec_idx_)
      {
        a = &a_;
        b = &b_;
        c = &c_;
        prec_idx = prec_idx_;
      }

      void bind_block_scales(const at::Tensor &scales_a_, const at::Tensor &scales_b_)
      {
        scales_a = &scales_a_;
        scales_b = &scales_b_;
      }
    };

    // One 64-bit seed from ATen's CPU generator when the round mode is SR,
    // and 0 otherwise, drawn exactly as cpu/gemm_backend.h draws it: on both
    // backends each output element's Philox stream is keyed on this seed and
    // the element's index, so an MPS result is the CPU's, stochastic rounding
    // included, and torch.manual_seed governs both. (The MPS generator would
    // give the device draws of its own, as CUDA's does, at the price of that
    // equality.)
    static LaunchContext make_context(bool use_rng, uint64_t draws_per_thread);

    // A NAIVE GEMM: single-family, palette, cross-family or odd.
    template <class Args>
    static void launch(const mptorch::gemm::GemmShape &s, const Args &args,
                       const LaunchContext &ctx)
    {
      launch_source(s, variant_source(args), uses_odd_kernel<Args>::value, ctx);
    }

    // KAHAN, BLOCK and TREE: the AccumulateArgs, with the algorithm, which
    // the kernel takes as a compile-time constant, as the host backends take
    // it as a template argument.
    template <class Base>
    static void launch(const mptorch::gemm::GemmShape &s,
                       const mptorch::gemm::AccumulateArgs<Base> &args, const LaunchContext &ctx)
    {
      launch_source(s, variant_source(args), uses_odd_kernel<Base>::value, ctx);
    }

    // The conv ops: the Args of the mac each runs, with the operands
    // gathered through the geometry.
    template <class Inner>
    static void launch(const mptorch::gemm::GemmShape &s, const mptorch::gemm::ConvArgs<Inner> &args,
                       const LaunchContext &ctx)
    {
      launch_conv(s, variant_source(args.inner), args.geom, uses_odd_kernel<Inner>::value, ctx);
    }

    // The block GEMM: the Args of its mac, over two packed operands. MPS has
    // no binary64, so the binary32 operand pair is the one the driver filled
    // (custom_matmul_block_entry.cpp refuses binary64 first).
    template <class Inner>
    static void launch(const mptorch::gemm::GemmShape &s,
                       const mptorch::gemm::BlockGemmArgs<Inner> &args, const LaunchContext &ctx)
    {
      launch_block(s, variant_source(args.inner), args.a, args.b, ctx);
    }

    // The Metal definition of `mpt_args()` for one op's Args: a function that
    // returns the struct with every field set by name, from the values the
    // schema arguments were packed into. One per Args type, in
    // gemm_backend.cpp.
    static std::string args_source(const mptorch::gemm::BinaryKSplitArgs &a);
    static std::string args_source(const mptorch::gemm::BinaryKSplitMixedArgs &a);
    static std::string args_source(const mptorch::gemm::BinaryKFusedArgs &a);
    static std::string args_source(const mptorch::gemm::BinaryKFusedMixedArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpSplitArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpSplitMixedArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpFusedArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpFusedMixedArgs &a);
    static std::string args_source(const mptorch::gemm::BinaryKSuperfpArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpBinaryKArgs &a);
    static std::string args_source(const mptorch::gemm::BlockSplitArgs &a);
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::BinaryKSplitArgs> &a);
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::BinaryKFusedArgs> &a);
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::SuperfpSplitArgs> &a);
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::SuperfpFusedArgs> &a);
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::BinaryKSuperfpArgs> &a);
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::SuperfpBinaryKArgs> &a);
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::BlockSplitArgs> &a);
    template <class Base>
    static std::string args_source(const mptorch::gemm::OddArgs<Base> &a);
    template <class Base>
    static std::string args_source(const mptorch::gemm::OddPaletteArgs<Base> &a);
    template <class Base>
    static std::string args_source(const mptorch::gemm::AccumulateArgs<mptorch::gemm::OddArgs<Base>> &a);

  private:
    // The instantiation's lines for an Args: its mpt_args(), and for an
    // AccumulateArgs the algorithm's line too.
    template <class Args>
    static std::string variant_source(const Args &args)
    {
      return args_source(args);
    }

    template <class Base>
    static std::string variant_source(const mptorch::gemm::AccumulateArgs<Base> &args)
    {
      return args_source(args) + accumulate_line(args.alg);
    }

    static std::string accumulate_line(AccumulateAlgorithm alg);

    static void launch_source(const mptorch::gemm::GemmShape &s, const std::string &args, bool odd,
                              const LaunchContext &ctx);
    static void launch_conv(const mptorch::gemm::GemmShape &s, const std::string &args,
                            const mptorch::gemm::ConvGeom &geom, bool odd, const LaunchContext &ctx);
    static void launch_block(const mptorch::gemm::GemmShape &s, const std::string &args,
                             const mptorch::gemm::BlockOperand &a, const mptorch::gemm::BlockOperand &b,
                             const LaunchContext &ctx);
  };

} // namespace mptorch::gemm_mps
