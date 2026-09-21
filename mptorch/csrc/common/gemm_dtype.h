#pragma once

// The two things the GEMM kernels need from the host that carry no ATen with
// them: the storage-dtype tag their operands arrive under, and the runtime
// to compile-time RoundMode dispatch.
//
// They live here rather than in common/dispatch.h so that a .cu file can name
// them without including <ATen/core/Tensor.h>. That header pulls in several
// hundred per-operator ATen headers and costs each .cu about 22 s of nvcc,
// for a declaration only the host prologue uses. dispatch.h includes it, so
// the ATen-side spellings (gemm_dtype_of, MPTORCH_DISPATCH_QUANT_TYPES) are
// still found through it.

#include "modes.h"
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mptorch
{

  // Whether an integer arriving over the torch.ops boundary names a
  // RoundMode. The Python wrappers only ever pass mptorch.number.RoundMode
  // values, but torch.ops.mptorch.* is a public entry point, and an integer
  // that names no mode must be rejected rather than silently mapped to
  // nearest-even by dispatch_round_mode's `default:` arm.
  //
  // The switch carries no `default:` on purpose: -Wswitch then names this
  // function when a mode is added to the enum, so the check cannot quietly
  // fall behind it. The range test first is not redundant: casting a value
  // the enum's underlying type cannot hold to a scoped enum is undefined
  // behaviour, and `rm` is whatever the caller passed.
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
      return true;
    }
    return false;
  }

  // The same question about an AccumulateAlgorithm, for the same reasons and
  // built the same way. The backends switch on the validated value to pick
  // an instantiation (their `launch` over a gemm::AccumulateArgs).
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

  // Turns a runtime RoundMode into a compile-time one: `f` is called with an
  // std::integral_constant naming the mode, so the policies it builds carry
  // a single cast body instead of a seven-way switch.
  //
  // Why a template parameter and not a runtime field: a `switch (round_mode)`
  // inside a GEMM's accumulate step puts all seven cast bodies (fourteen for
  // a split mac, which rounds twice per step) into the K-loop, which is more
  // instruction memory than the loop can fetch from and prevents its full
  // unrolling. That cost 1.2-2.3x of kernel time. With the mode fixed at
  // compile time one body is live and the loop unrolls. The price is one
  // instantiation of the caller's body per mode, which is host code that
  // builds a policy struct and launches; the kernels behind them are what
  // multiplies, and taking the storage dtype out of the kernel template (see
  // GemmDtype below) made room for that. The elementwise quantizers' entry
  // points do the same switch by hand; the GEMM's eight get it from here.
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
      // The GEMM's callers reach this only through check_matmul_inputs,
      // which rejects anything is_round_mode() does not name; the `default:`
      // exists because `rm` is an enum holding whatever the caller cast.
      f(std::integral_constant<RoundMode, RoundMode::RNE>{});
      break;
    }
  }

  // The storage dtype a GEMM's operands arrive under. The kernels take their
  // operands as `const void *` plus one of these instead of being
  // instantiated once per storage dtype, because `scalar_t` was never
  // anything but the load/store type: the three instantiations of a GEMM
  // differed only in two tile loads per K-step and one store per output
  // element, and tripled everything else (the cast bodies, the unrolled
  // K-loop, the compile time and the SASS). Switching on the tag at load
  // time runs the same `static_cast<float>` the template would have, chosen
  // at runtime rather than at compile time, so the values are identical. The
  // elementwise quantizers keep the per-dtype instantiation
  // (MPTORCH_DISPATCH_QUANT_TYPES) because loading and storing is all they
  // do, and they vectorize per dtype through SIMDTraits.
  //
  // Double is the exception, because it is a different carrier rather than a
  // different load: a float64 GEMM computes in binary64 (bit_helper.h's
  // carrier_t), so the tag selects which kernel runs, on the host in the
  // backend's launch(), and never reaches a float kernel's load switch.
  enum class GemmDtype : int
  {
    Float = 0,
    Half = 1,
    BFloat16 = 2,
    Double = 3,
  };

} // namespace mptorch
