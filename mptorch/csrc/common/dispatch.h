#pragma once

#include "gemm_dtype.h"
#include "modes.h"
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

// Type dispatch for the GEMM and elementwise quantize entry points.
//
// Both ops dispatch on the storage dtype and compute in its carrier
// (carrier_t, bit_helper.h): binary64 for a float64 tensor, binary32 for the
// other three, whose values binary32 holds exactly. Computing a float64
// tensor in double rather than narrowing it to float first is what lets an
// input be rounded once, directly, and lets a format finer or wider than
// binary32 be reached at all.
//
// The GEMM's half of the dispatch, mptorch::GemmDtype and
// dispatch_round_mode, lives in common/gemm_dtype.h, which carries no ATen,
// so the .cu files can name them without paying for <ATen/core/Tensor.h>.
// This header includes it, so every spelling resolves from here.
#define MPTORCH_DISPATCH_QUANT_TYPES(TYPE, NAME, ...)       \
  AT_DISPATCH_SWITCH(                                       \
      TYPE, NAME,                                           \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
      AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

namespace mptorch
{

  // The dtype tag of a GEMM's operand pair, with the checks that a typed
  // `data_ptr<scalar_t>()` used to make for free: operands that disagree, or
  // a dtype the kernel cannot load, are rejected rather than reinterpreted
  // through a `const void *`. A (float64, float32) pair is one of those: the
  // two carriers are different kernels, and picking one for the pair would
  // round the other operand in a carrier it did not ask for.
  //
  // A build with MPTORCH_NO_FP64 has no binary64 GEMM kernels to launch, and
  // says so here, before the RNG state is drawn, rather than narrowing in
  // silence to a result the default build would not give.
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
    case at::kDouble:
#if defined(MPTORCH_NO_FP64)
      TORCH_CHECK(false, op_name, ": this build has no float64 GEMM kernels (it was built with "
                                  "MPTORCH_NO_FP64=1); pass float32 operands, or rebuild without it");
#else
      return GemmDtype::Double;
#endif
    default:
      TORCH_CHECK(false, op_name, ": expected float32, float64, float16 or bfloat16 operands, got ",
                  a.scalar_type());
    }
  }

} // namespace mptorch
