// _carrier_odd on the CPU: common/carrier_odd.h's host primitives applied
// elementwise, so the tests can see what the format casts after them erase
// (zero signs, the largest finite value, subnormal parity). A test op: small
// tensors, one thread.

#include "../common/carrier_odd_host.h"
#include "../quant_ops.h"
#include <ATen/ops/empty_like.h>

using at::Tensor;

#if !defined(MPTORCH_NO_ODD)
namespace
{
  template <class T>
  void run(const T *a, const T *b, const T *c, T *o, int64_t n, int64_t op)
  {
    namespace odd = mptorch::gemm::odd;
    for (int64_t i = 0; i < n; ++i)
      o[i] = op == 0 ? odd::fma(a[i], b[i], c[i]) : op == 1 ? odd::add(a[i], b[i])
                                                            : odd::mul(a[i], b[i]);
  }
} // namespace
#endif

Tensor carrier_odd_cpu(Tensor a, Tensor b, Tensor c, int64_t op)
{
#if defined(MPTORCH_NO_ODD)
  mptorch::gemm::odd_unavailable("_carrier_odd");
#else
  mptorch::gemm::check_carrier_odd(a, b, c, op);
  Tensor a_c = a.contiguous(), b_c = b.contiguous(), c_c = c.contiguous();
  Tensor o = at::empty_like(a_c);
  if (a.scalar_type() == at::kFloat)
    run(a_c.data_ptr<float>(), b_c.data_ptr<float>(), c_c.data_ptr<float>(),
        o.data_ptr<float>(), o.numel(), op);
  else
    run(a_c.data_ptr<double>(), b_c.data_ptr<double>(), c_c.data_ptr<double>(),
        o.data_ptr<double>(), o.numel(), op);
  return o;
#endif
}
