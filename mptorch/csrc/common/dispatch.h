#pragma once

#include "gemm_dtype.h"
#include "modes.h"
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

// Type dispatch for the GEMM and elementwise quantize entry points.
//
// The elementwise quantizers dispatch on the storage dtype and round in its
// carrier (carrier_t, bit_helper.h): binary64 for a float64 tensor, binary32
// for the other three, whose values binary32 holds. A float64 tensor used to
// be narrowed to float32 up front instead (finding G6), because every kernel
// converted to float on load anyway, so a double instantiation computed
// nothing a float one did not. Rounding in double is what changes that: the
// input is rounded once, directly, and a format finer or wider than binary32
// can be reached (dev/binary64_carrier_plan.md, phase 3).
//
// The GEMM still narrows a float64 pair (narrow_float64 below, called from
// common/gemm_host.h) until it has double kernels of its own (phase 4). Its
// half of the dispatch -- mptorch::GemmDtype and dispatch_round_mode -- lives
// in common/gemm_dtype.h, which carries no ATen, so the .cu files can name
// them without paying for <ATen/core/Tensor.h> (finding H1). This header
// includes it, so every existing spelling still resolves from here.
#define MPTORCH_DISPATCH_QUANT_TYPES(TYPE, NAME, ...)       \
  AT_DISPATCH_SWITCH(                                       \
      TYPE, NAME,                                           \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
      AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)   \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

namespace mptorch
{

  // The check that `data_ptr<scalar_t>()` used to make for free: a GEMM whose
  // operands disagree, or whose dtype the kernel cannot load, must be rejected
  // rather than reinterpreted.
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


  // The GEMM's float64 path until phase 4: narrows a float64 operand pair to
  // float32 in place and returns whether the result has to be widened back.
  // Anything else is left untouched.
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
