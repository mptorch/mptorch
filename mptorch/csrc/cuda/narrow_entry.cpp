// The CUDA entry point of narrow_float64: the tensors and the stream, and
// nothing nvcc has to see (narrow_kernel.h says why). A .cpp inside csrc/cuda/,
// so a CPU-only build skips it with the .cu it drives.

#include "../common/narrow_host.h"
#include "../quant_ops.h"
#include "narrow_kernel.h"
#include "vector_load.h"
#include <ATen/cuda/CUDAContext.h>

using at::Tensor;

Tensor narrow_float64_cuda(Tensor a, c10::ScalarType dtype)
{
  auto [a_c, o] = mptorch::narrow_float64_tensors(a, dtype);
  const int64_t size = a_c.numel();
  if (size == 0)
    return o;
  // the kernel loads two doubles at a time, 16 bytes aligned; mptorch.quant
  // only ever hands this a kernel's own output, which already is
  a_c = mptorch::vector_loadable(a_c);
  using mptorch::narrow_cuda::Target;
  const Target target = dtype == c10::ScalarType::Half       ? Target::Float16
                        : dtype == c10::ScalarType::BFloat16 ? Target::BFloat16
                                                             : Target::Float32;
  mptorch::narrow_cuda::launch(a_c.data_ptr<double>(), o.data_ptr(), size, target,
                               at::cuda::getCurrentCUDAStream());
  return o;
}
