#pragma once

// The two things the GEMM kernels need from the host that need no ATen: the
// tag naming the storage dtype their operands arrive in, and the conversion
// of a runtime RoundMode into a compile-time one.
//
// They are here rather than in common/dispatch.h so that a .cu file can use
// them without including <ATen/core/Tensor.h>. That header pulls in several
// hundred per-operator ATen headers and costs each .cu about 22 s of nvcc
// time, for declarations only the host-side driver uses. dispatch.h includes
// this header, so its ATen-side names (gemm_dtype_of,
// MPTORCH_DISPATCH_QUANT_TYPES) and these are all found through it.

#include "modes.h"
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mptorch
{

  // Whether an integer arriving over the torch.ops boundary names a
  // RoundMode. The Python wrappers only ever pass mptorch.number.RoundMode
  // values, but torch.ops.mptorch.* can be called directly, and an integer
  // that names no mode must be rejected, not silently treated as RNE by
  // dispatch_round_mode's `default:` arm.
  //
  // The switch has no `default:` on purpose: -Wswitch then points at this
  // function when a mode is added to the enum, so the check cannot fall
  // behind the enum unnoticed. The range test before it is needed: casting a
  // value outside the range of the enum's underlying type to a scoped enum is
  // undefined behaviour, and `rm` is whatever the caller passed.
  inline bool is_round_mode(int64_t rm)
  {
    using U = std::underlying_type_t<RoundMode>;
    if (rm < static_cast<int64_t>(std::numeric_limits<U>::min()) ||
        rm > static_cast<int64_t>(std::numeric_limits<U>::max()))
      return false;
    switch (static_cast<RoundMode>(rm))
    {
    case RoundMode::RNE:
    case RoundMode::RNA:
    case RoundMode::RU:
    case RoundMode::RD:
    case RoundMode::RZ:
    case RoundMode::RO:
    case RoundMode::SR:
    case RoundMode::SR_B:
    case RoundMode::SR_C:
      return true;
    }
    return false;
  }

  // The same check for an AccumulateAlgorithm, for the same reasons and
  // written the same way. The backends switch on the validated value to
  // choose an instantiation (their `launch` over a gemm::AccumulateArgs).
  inline bool is_accumulate_algorithm(int64_t alg)
  {
    using U = std::underlying_type_t<AccumulateAlgorithm>;
    if (alg < static_cast<int64_t>(std::numeric_limits<U>::min()) ||
        alg > static_cast<int64_t>(std::numeric_limits<U>::max()))
      return false;
    switch (static_cast<AccumulateAlgorithm>(alg))
    {
    case AccumulateAlgorithm::NAIVE:
    case AccumulateAlgorithm::KAHAN:
    case AccumulateAlgorithm::BLOCK:
    case AccumulateAlgorithm::TREE:
      return true;
    }
    return false;
  }

  // Converts a runtime RoundMode into a compile-time one: `f` is called with
  // an std::integral_constant naming the mode, so the policies it builds
  // hold one cast body instead of a switch over the seven modes (RNE, RNA,
  // RU, RD, RZ, RO and SR).
  //
  // Why a template parameter rather than a runtime field: a
  // `switch (round_mode)` inside a GEMM's accumulate step puts all seven cast
  // bodies (fourteen for a split mac, which rounds twice per step) into the
  // K-loop, which is more code than the loop's instruction fetch can serve,
  // and prevents unrolling the loop fully. That cost 1.2-2.3x in kernel time.
  // With the mode fixed at compile time one body is live and the loop
  // unrolls. The price is one instantiation of the caller's body per mode.
  // The caller is host code that builds a policy and launches; what
  // multiplies is the kernels behind it, and taking the storage dtype out of
  // the kernel template (GemmDtype below) made room for that. The elementwise
  // quantizers' entry points switch by hand; the GEMM and block-format
  // kernels' launches call this function.
  template <class F>
  void dispatch_round_mode(RoundMode rm, F &&f)
  {
    switch (rm)
    {
    case RoundMode::RNA:
      f(std::integral_constant<RoundMode, RoundMode::RNA>{});
      break;
    case RoundMode::RU:
      f(std::integral_constant<RoundMode, RoundMode::RU>{});
      break;
    case RoundMode::RD:
      f(std::integral_constant<RoundMode, RoundMode::RD>{});
      break;
    case RoundMode::RZ:
      f(std::integral_constant<RoundMode, RoundMode::RZ>{});
      break;
    case RoundMode::RO:
      f(std::integral_constant<RoundMode, RoundMode::RO>{});
      break;
    case RoundMode::SR:
      f(std::integral_constant<RoundMode, RoundMode::SR>{});
      break;
    case RoundMode::RNE:
    default:
      // Callers validate the mode first (the GEMMs in check_matmul_inputs,
      // with is_round_mode()), and must have turned SR_B and SR_C into SR
      // (stochastic_host.h), which would otherwise round to nearest even
      // here. The `default:` exists because `rm` holds whatever integer the
      // caller cast to the enum.
      f(std::integral_constant<RoundMode, RoundMode::RNE>{});
      break;
    }
  }

  // The storage dtype a GEMM's operands arrive in. The kernels take their
  // operands as `const void *` plus one of these tags, instead of being
  // instantiated once per storage dtype, because `scalar_t` was never more
  // than the type operands are loaded and stored as: the three
  // instantiations of a GEMM differed only in two tile loads per K-step and
  // one store per output element, and tripled everything else (the cast
  // bodies, the unrolled K-loop, the compile time and the machine code).
  // Switching on the tag at load time runs the same `static_cast<float>` the
  // template ran, chosen at run time instead of compile time, so the values
  // are identical. The elementwise quantizers keep one instantiation per
  // dtype (MPTORCH_DISPATCH_QUANT_TYPES), because loading and storing is all
  // they do, and they vectorize per dtype through SIMDTraits.
  //
  // Double is the exception, because it is a different carrier rather than
  // a different load: a float64 GEMM computes in binary64 (bit_helper.h's
  // carrier_t), so the backend's launch() uses the tag on the host to choose
  // which kernel runs, and it never reaches a binary32 kernel's load switch.
  enum class GemmDtype : int
  {
    Float = 0,
    Half = 1,
    BFloat16 = 2,
    Double = 3,
  };

} // namespace mptorch
