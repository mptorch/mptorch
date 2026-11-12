// The CPU kernel of _stochastic_cast (common/stochastic_cast_host.h): a
// format's stochastic cast with each element's random word given, for
// tests/test_stochastic_rounding.py. Test-only.

#include "../common/stochastic_cast_host.h"
#include "../quant_ops.h"
#include <ATen/ops/empty_like.h>

using at::Tensor;

namespace
{
  template <class T, class W>
  void run(const mptorch::stochastic_cast::Job<T> &j, const T *x, const int64_t *draws, T *o, int64_t n)
  {
    for (int64_t i = 0; i < n; ++i)
      o[i] = mptorch::stochastic_cast::cast(j, x[i], static_cast<W>(static_cast<uint64_t>(draws[i])));
  }
} // namespace

Tensor stochastic_cast_cpu(Tensor x, Tensor draws, int64_t family, c10::IntArrayRef format,
                           int64_t stochastic_mode)
{
  namespace sc = mptorch::stochastic_cast;
  sc::check_tensors(x, draws);
  Tensor x_c = x.contiguous(), d_c = draws.contiguous();
  Tensor o = at::empty_like(x_c);
  if (x.scalar_type() == at::kFloat)
    run<float, uint32_t>(sc::make_job<float>(family, format, stochastic_mode), x_c.data_ptr<float>(),
                         d_c.data_ptr<int64_t>(), o.data_ptr<float>(), o.numel());
  else
    run<double, uint64_t>(sc::make_job<double>(family, format, stochastic_mode), x_c.data_ptr<double>(),
                          d_c.data_ptr<int64_t>(), o.data_ptr<double>(), o.numel());
  return o;
}
