#pragma once

// What the CPU and CUDA entry points of narrow_float64 share: the argument
// checks, and the tensors they read and write. The rounding itself is in
// narrow_binary64.h.

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>

namespace mptorch
{

  // Returns (input, output) for a narrowing kernel: the input made
  // contiguous, and a fresh tensor of `dtype` shaped like it. The kernels
  // walk data_ptr() linearly, so a strided input is copied first, as the
  // elementwise quantizers do. Rejects a non-float64 input and a target
  // dtype other than float32, float16 or bfloat16.
  inline std::pair<at::Tensor, at::Tensor> narrow_float64_tensors(const at::Tensor &a,
                                                                  c10::ScalarType dtype)
  {
    TORCH_CHECK(a.scalar_type() == c10::ScalarType::Double,
                "narrow_float64: the input must be float64, got ", a.scalar_type());
    TORCH_CHECK(dtype == c10::ScalarType::Half || dtype == c10::ScalarType::BFloat16 ||
                    dtype == c10::ScalarType::Float,
                "narrow_float64: narrows to float32, float16 or bfloat16, got ", dtype);
    at::Tensor a_c = a.contiguous();
    at::Tensor o = at::empty(a_c.sizes(), a_c.options().dtype(dtype));
    return {a_c, o};
  }

} // namespace mptorch
