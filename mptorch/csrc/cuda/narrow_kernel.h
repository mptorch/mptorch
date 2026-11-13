#pragma once

// The device half of narrow_float64, as its entry point sees it: one launch
// over raw pointers. As with the GEMM (cuda/gemm_backend.h), the .cu that
// defines it includes no ATen headers, which keeps that object's fixed nvcc
// cost near 3 s rather than about 25 s; the tensors and the stream stay in
// narrow_entry.cpp.

#include <cuda_runtime_api.h>
#include <cstdint>

namespace mptorch::narrow_cuda
{

  // The storage dtype the float64 input is rounded onto.
  enum class Target
  {
    Float32,
    Float16,
    BFloat16,
  };

  // narrow_kernel.cu: rounds `size` float64 values at `a` onto `target`
  // (round-to-nearest-even, once) and writes the resulting words to `o`, on
  // `stream`. `a` must be 16-byte aligned; `o` is typed by the target.
  void launch(const double *a, void *o, int64_t size, Target target, cudaStream_t stream);

} // namespace mptorch::narrow_cuda
