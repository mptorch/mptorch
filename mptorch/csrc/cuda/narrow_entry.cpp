// The CUDA entry point of narrow_float64: the tensors and the stream, and
// nothing nvcc has to see (narrow_kernel.h says why). A .cpp inside csrc/cuda/,
// so a CPU-only build skips it with the .cu it drives.

#include "../common/narrow_host.h"
#include "../quant_ops.h"
#include "narrow_kernel.h"
#include <ATen/cuda/CUDAContext.h>

using at::Tensor;

Tensor narrow_float64_cuda(Tensor a, c10::ScalarType dtype)
{
  auto [a_c, o] = mptorch::narrow_float64_tensors(a, dtype);
  const int64_t size = a_c.numel();
  if (size == 0)
    return o;
  // The kernel loads two doubles at a time as a 16-byte aligned vector, and a
  // contiguous view need not start on a 16-byte boundary: `x[1:]` of a float64
  // tensor is 8 bytes in, and loading it is a misaligned-address fault that
  // poisons the context. A fresh allocation is aligned, so such a view is
  // copied; mptorch.quant only ever hands this a kernel's own output.
  if (reinterpret_cast<uintptr_t>(a_c.data_ptr()) % 16 != 0)
    a_c = a_c.clone();
  using mptorch::narrow_cuda::Target;
  const Target target = dtype == c10::ScalarType::Half       ? Target::Float16
                        : dtype == c10::ScalarType::BFloat16 ? Target::BFloat16
                                                             : Target::Float32;
  mptorch::narrow_cuda::launch(a_c.data_ptr<double>(), o.data_ptr(), size, target,
                               at::cuda::getCurrentCUDAStream());
  return o;
}
