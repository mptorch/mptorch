#pragma once

#include "gemm_dtype.h"
#include "modes.h"
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

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
//
// The GEMM's half of this -- mptorch::GemmDtype and dispatch_round_mode --
// moved to common/gemm_dtype.h, which carries no ATen, so the .cu files can
// name them without paying for <ATen/core/Tensor.h> (finding H1). This header
// includes it, so every existing spelling still resolves from here.
#define MPTORCH_DISPATCH_QUANT_TYPES(TYPE, NAME, ...)      \
  AT_DISPATCH_SWITCH(                                      \
      TYPE, NAME,                                          \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
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
