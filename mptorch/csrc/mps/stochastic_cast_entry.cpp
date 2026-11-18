// The MPS entry point of _stochastic_cast (common/stochastic_cast_host.h), a
// test-only op with no Metal kernel, so it only raises. Its CPU and CUDA
// entry points are cpu/stochastic_cast_kernel.cpp and
// cuda/stochastic_cast_entry.cpp. tests/test_stochastic_rounding.py checks
// the CPU's stochastic casts against P3109 through this op, and
// tests/test_mps.py compares every MPS stochastic op with the CPU's, word for
// word.

#include "../quant_ops.h"
#include <c10/util/Exception.h>

using at::Tensor;

Tensor stochastic_cast_mps(Tensor, Tensor, int64_t, c10::IntArrayRef, int64_t)
{
  TORCH_CHECK(
      false, "_stochastic_cast: a test-only op with no MPS kernel; run it on a CPU or CUDA tensor");
}
