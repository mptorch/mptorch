#pragma once

// The MPS backend as common/gemm_host.h's driver sees it: a launch context
// and one `launch` per Args type, the shape of cpu/gemm_backend.h and
// cuda/gemm_backend.h. What differs is where the policies are built. The CPU
// and CUDA backends build them here, on the host, and hand the kernel the
// result; this one hands the kernel the Args themselves, written out as
// Metal source (args_source below), and the kernel builds the policies from
// them with the Args' own factories: with_accumulator, as the other backends
// do, and on the mixed ops with_slot, one element's slot at a time, where
// they take with_palette's whole palette (gemm.metal says why). That makes
// every format constant a compile-time constant of the kernel, at the price
// of one compile per distinct format (metal_runtime.h).
//
// binary32 only: MPS has no float64 tensors, so there is no binary64 kernel
// to pick and no MPTORCH_NO_FP64 to honour.

#include "../common/gemm_args.h"
#include <ATen/core/Tensor.h>
#include <cstdint>
#include <string>

namespace mptorch::gemm_mps
{

  struct MpsBackend
  {
    // The SR seed, drawn like the CPU backend's, and the tensors, which the
    // driver binds (gemm_host.h's bind_tensors) because a kernel takes an MPS
    // tensor as its MTLBuffer and a byte offset rather than as the pointer
    // GemmShape carries. They are the driver's locals and outlive the launch.
    struct LaunchContext
    {
      uint64_t seed = 0;
      const at::Tensor *a = nullptr;
      const at::Tensor *b = nullptr;
      const at::Tensor *c = nullptr;
      const at::Tensor *prec_idx = nullptr;

      void bind(const at::Tensor &a_, const at::Tensor &b_, const at::Tensor &c_,
                const at::Tensor *prec_idx_)
      {
        a = &a_;
        b = &b_;
        c = &c_;
        prec_idx = prec_idx_;
      }
    };

    // One 64-bit seed from ATen's CPU generator when the round mode is SR,
    // and 0 otherwise, exactly as cpu/gemm_backend.h draws it: each output
    // element's Philox stream is keyed on this seed and the element's index
    // on both backends, so an MPS result is the CPU's, stochastic rounding
    // included, and torch.manual_seed governs both. (The MPS generator would
    // make the device's draws its own, as CUDA's are, at the price of that
    // equality.)
    static LaunchContext make_context(bool use_rng, uint64_t draws_per_thread);

    template <class Args>
    static void launch(const mptorch::gemm::GemmShape &s, const Args &args,
                       const LaunchContext &ctx)
    {
      launch_source(s, args_source(args), ctx);
    }

    // The Metal definition of `mpt_args()` for one op's Args: a function
    // returning the struct with every field set by name, from the values
    // the schema packed. One per Args type, in gemm_backend.cpp.
    static std::string args_source(const mptorch::gemm::BinaryKSplitArgs &a);
    static std::string args_source(const mptorch::gemm::BinaryKSplitMixedArgs &a);
    static std::string args_source(const mptorch::gemm::BinaryKFusedArgs &a);
    static std::string args_source(const mptorch::gemm::BinaryKFusedMixedArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpSplitArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpSplitMixedArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpFusedArgs &a);
    static std::string args_source(const mptorch::gemm::SuperfpFusedMixedArgs &a);

  private:
    static void launch_source(const mptorch::gemm::GemmShape &s, const std::string &args,
                              const LaunchContext &ctx);
  };

} // namespace mptorch::gemm_mps
