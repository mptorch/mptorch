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
#include <type_traits>

namespace mptorch
{

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
    default: // RoundMode::RNE
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
  enum class GemmDtype : int
  {
    Float = 0,
    Half = 1,
    BFloat16 = 2,
  };

} // namespace mptorch
