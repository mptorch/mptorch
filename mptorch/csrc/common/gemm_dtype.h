#pragma once

// The two things the GEMM kernels need from the host that carry no ATen with
// them: the storage-dtype tag its operands arrive under, and the runtime
// -> compile-time RoundMode dispatch.
//
// They live here rather than in common/dispatch.h so a .cu can name them
// without including <ATen/core/Tensor.h>, which drags ~420 <ATen/ops/*.h>
// headers behind it and costs each .cu ~22 s of nvcc for a declaration only
// the host prologue uses. dispatch.h includes this file, so the ATen-side
// spellings (gemm_dtype_of, narrow_float64) are still found where they always
// were. See dev/gemm_roadmap.md (finding H1).

#include "modes.h"
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mptorch
{

  // Whether an integer arriving over the torch.ops boundary names a
  // RoundMode. The Python wrappers only ever pass mptorch.number.RoundMode
  // values, but torch.ops.mptorch.* is a public entry point and an unnamed
  // integer used to fall through dispatch_round_mode's `default:` and round
  // to nearest-even in silence.
  //
  // The switch carries no `default:` on purpose: -Wswitch then names this
  // function when a mode is added to the enum, so the check cannot quietly
  // fall behind it. The range test first is not redundant -- casting a value
  // its underlying type cannot hold to a scoped enum is undefined, and `rm`
  // is whatever the caller passed.
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

  // Turns a runtime RoundMode into a compile-time one (finding K1): `f` is
  // called with an std::integral_constant naming the mode, so the policies it
  // builds carry a single cast body instead of a seven-way switch. This is
  // what the elementwise quantizers' own entry points already did by hand
  // (see cuda/binaryK_kernel.cu's launch_kernels); the GEMM's eight entry
  // points get it from here rather than each repeating the switch.
  //
  // It costs one instantiation of the caller's body per mode, all seven of
  // which are host code that builds a policy struct and launches -- the
  // kernels behind them are the only thing that multiplies, and K2 divided
  // that count by three first, which is why the pair ships together.
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
      // is here because `rm` is an enum holding whatever the caller cast.
      f(std::integral_constant<RoundMode, RoundMode::RNE>{});
      break;
    }
  }

  // The GEMM kernels take their operands as `const void *` plus one of these
  // instead of being instantiated once per storage dtype (finding K2).
  // `scalar_t` was never anything but the load/store type, so the three
  // instantiations of a GEMM differed in two tile loads per 16 K-steps and one
  // store per output element, and tripled everything else -- the cast bodies,
  // the unrolled K-loop, the compile time and the SASS. The same
  // `static_cast<float>` still runs; which one is chosen at runtime rather
  // than at compile time, so the values are identical. The elementwise
  // quantizers keep MPTORCH_DISPATCH_QUANT_TYPES: they *are* the load/store,
  // and they vectorize per dtype (SIMDTraits).
  //
  // Double is the exception, because it is a different carrier rather than a
  // different load: a float64 GEMM computes in binary64 (bit_helper.h's
  // carrier_t), so the tag selects which kernel runs, on the host, before
  // launch -- the backend's launch() -- and never reaches a float kernel's
  // load switch (dev/binary64_carrier_plan.md, phase 4).
  enum class GemmDtype : int
  {
    Float = 0,
    Half = 1,
    BFloat16 = 2,
    Double = 3,
  };

} // namespace mptorch
