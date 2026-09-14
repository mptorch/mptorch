#pragma once

// The device half of narrow_float64, as its entry point sees it: one launch
// over raw pointers. Like the GEMM's (cuda/gemm_backend.h), the .cu that
// defines it includes no ATen, which keeps that object's fixed cost near 3 s
// rather than ~25 s (finding H1); the tensors and the stream stay in
// narrow_entry.cpp.

#include <cuda_runtime_api.h>
#include <cstdint>

namespace mptorch::narrow_cuda
{

  enum class Target
  {
    Float32,
    Float16,
    BFloat16,
  };

  // narrow_kernel.cu: rounds `size` float64 values at `a` onto `target`,
  // writing its words to `o`, on `stream`.
  void launch(const double *a, void *o, int64_t size, Target target, cudaStream_t stream);

} // namespace mptorch::narrow_cuda
