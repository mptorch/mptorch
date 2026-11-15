// _carrier_odd on an Apple GPU (phase Z): common/carrier_odd_metal.h's
// binary32 primitives applied elementwise by mps/carrier_odd.metal, which
// tests/test_carrier_rounding.py holds to an exact reference and to the
// CPU's op word for word. A test op: MPS has no float64, so binary32 only.

#include "../common/carrier_odd_host.h"
#include "../quant_ops.h"
#include "metal_runtime.h"
#include <ATen/ops/empty_like.h>
#include <limits>
#include <string>

using at::Tensor;

Tensor carrier_odd_mps(Tensor a, Tensor b, Tensor c, int64_t op)
{
#if defined(MPTORCH_NO_ODD)
  mptorch::gemm::odd_unavailable("_carrier_odd");
#else
  mptorch::gemm::check_carrier_odd(a, b, c, op);
  Tensor a_c = a.contiguous(), b_c = b.contiguous(), c_c = c.contiguous();
  Tensor o = at::empty_like(a_c);
  if (o.numel() == 0)
    return o;
  TORCH_CHECK(o.numel() <= std::numeric_limits<uint32_t>::max(),
              "_carrier_odd: an MPS call takes at most 2^32 - 1 elements, got ", o.numel());
  mptorch_mps::launch(mptorch_mps::KernelSource::CarrierOdd,
                      "#define MPT_ODD_OP " + std::to_string(op) + "\n", {a_c, b_c, c_c, o},
                      static_cast<uint64_t>(o.numel()));
  return o;
#endif
}
