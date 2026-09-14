#pragma once

// What the CPU and CUDA entry points of narrow_float64 share: the checks, and
// the tensors they read and write. The rounding itself is narrow_binary64.h.

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>

namespace mptorch
{

  // The contiguous input a narrowing kernel reads, and the output it writes,
  // shaped like the input: data_ptr() walks storage linearly, so a strided
  // input is copied first, as the elementwise quantizers do.
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
