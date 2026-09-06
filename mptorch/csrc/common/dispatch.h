#pragma once

#include "modes.h"
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <type_traits>

// Type dispatch for the GEMM and elementwise quantize entry points.
//
// scalar_t is only ever these kernels' *load/store* type: every one of them
// converts to float on load (custom_matmul_kernel's tile buffers, the
// Accumulator and the whole Mac policy chain are float; the elementwise
// quantizers' eval() takes a float) and converts back on store. So a
// float64 operand was never computed in double -- it was narrowed on load
// and widened on store, one element at a time, by a kernel compiled a
// second time for no numerical gain.
//
// Narrowing the tensors up front instead is value-identical (the same
// double -> float RNE conversion, just done in a cast pass rather than in
// the load) and lets the dispatch drop at::ScalarType::Double, which is a
// quarter of the GEMM's 52 kernel instantiations and of the compile time
// and .nv_fatbin that go with them. See dev/gemm_perf_audit.md (finding G6).
#define MPTORCH_DISPATCH_QUANT_TYPES(TYPE, NAME, ...)      \
  AT_DISPATCH_SWITCH(                                      \
      TYPE, NAME,                                          \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

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
  // instead of being instantiated once per storage dtype (finding K2). The
  // paragraph above is the whole argument: `scalar_t` was never anything but
  // the load/store type, so the three instantiations of a GEMM differed in two
  // tile loads per 16 K-steps and one store per output element, and tripled
  // everything else -- the cast bodies, the unrolled K-loop, the compile time
  // and the SASS. The same `static_cast<float>` still runs; which one is
  // chosen at runtime rather than at compile time, so the values are
  // identical. The elementwise quantizers keep the macro: they *are* the
  // load/store, and they vectorize per dtype (SIMDTraits).
  enum class GemmDtype : int
  {
    Float = 0,
    Half = 1,
    BFloat16 = 2,
  };

  // Also the check that `data_ptr<scalar_t>()` used to make for free: a
  // GEMM whose operands disagree, or whose dtype the kernel cannot load,
  // must be rejected rather than reinterpreted.
  inline GemmDtype gemm_dtype_of(const at::Tensor &a, const at::Tensor &b, const char *op_name)
  {
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), op_name,
                ": both operands must have the same dtype, got ", a.scalar_type(), " and ",
                b.scalar_type());
    switch (a.scalar_type())
    {
    case at::kFloat:
      return GemmDtype::Float;
    case at::kHalf:
      return GemmDtype::Half;
    case at::kBFloat16:
      return GemmDtype::BFloat16;
    default:
      TORCH_CHECK(false, op_name, ": expected float32, float16 or bfloat16 operands, got ",
                  a.scalar_type());
    }
  }


  // Companion to MPTORCH_DISPATCH_QUANT_TYPES: narrows float64 operands to
  // float32 in place and returns whether the result has to be widened back.
  // Anything else is left untouched.
  inline bool narrow_float64(at::Tensor &a)
  {
    if (a.scalar_type() != at::kDouble)
      return false;
    a = a.to(at::kFloat);
    return true;
  }

  inline bool narrow_float64(at::Tensor &a, at::Tensor &b)
  {
    // Both, so a mismatched pair still reaches the dispatch and is rejected
    // there rather than being quietly made to agree.
    if (a.scalar_type() != at::kDouble || b.scalar_type() != at::kDouble)
      return false;
    a = a.to(at::kFloat);
    b = b.to(at::kFloat);
    return true;
  }

  inline at::Tensor widen_float64(at::Tensor t, bool widen)
  {
    return widen ? t.to(at::kDouble) : t;
  }

} // namespace mptorch
