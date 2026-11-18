#pragma once

#include "gemm_dtype.h"
#include "modes.h"
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

// Type dispatch for the entry points of the elementwise quantizers and the
// GEMMs.
//
// Both dispatch on the storage dtype and compute in its carrier, the IEEE
// format the arithmetic runs in (carrier_t, bit_helper.h): binary64 for a
// float64 tensor, and binary32 for the other three dtypes, whose values
// binary32 holds exactly. Computing a float64 tensor in binary64 rather than
// narrowing it to binary32 first means each input is rounded only once,
// directly into the target format, and makes formats finer or wider than
// binary32 reachable at all.
//
// The GEMM's half of the dispatch, mptorch::GemmDtype and
// dispatch_round_mode, is in common/gemm_dtype.h, which includes no ATen
// header, so that the .cu files can use them without paying for
// <ATen/core/Tensor.h>. This header includes gemm_dtype.h, so all of these
// names are available from here.
#define MPTORCH_DISPATCH_QUANT_TYPES(TYPE, NAME, ...)       \
  AT_DISPATCH_SWITCH(                                       \
      TYPE, NAME,                                           \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
      AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

namespace mptorch
{

  // The dtype tag of a GEMM's pair of operands. It makes the checks a typed
  // `data_ptr<scalar_t>()` would make: operands whose dtypes differ, and a
  // dtype the kernel cannot load, are rejected rather than read through a
  // `const void *` as something they are not. A (float64, float32) pair is
  // one of those: the two carriers are different kernels, and picking one for
  // the pair would round the other operand in a carrier it did not ask for.
  //
  // A build with MPTORCH_NO_FP64 has no binary64 GEMM kernels, and says so
  // here, before the RNG state is drawn, rather than silently narrowing to a
  // result the default build would not give.
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
