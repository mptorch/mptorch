// narrow_float64 on the CPU: common/narrow_binary64.h over the tensor, on
// ATen's thread pool like the elementwise quantizers (utils.h).

#include "../common/narrow_binary64.h"
#include "../common/narrow_host.h"
#include "../quant_ops.h"
#include "utils.h"
#include <cstring>

using at::Tensor;

namespace
{

  template <class Word, int ExpBits, int ManBits>
  void narrow_run(const double *a, Word *o, int64_t size)
  {
    at::parallel_for(0, size, mptorch_cpu::quant_grain_size,
                     [=](int64_t begin, int64_t end)
                     {
                       for (int64_t i = begin; i < end; ++i)
                       {
                         uint64_t w;
                         std::memcpy(&w, a + i, sizeof w);
                         o[i] = narrow_binary64<Word, ExpBits, ManBits>(w);
                       }
                     });
  }

} // namespace

Tensor narrow_float64_cpu(Tensor a, c10::ScalarType dtype)
{
  auto [a_c, o] = mptorch::narrow_float64_tensors(a, dtype);
  const int64_t size = a_c.numel();
  if (size == 0)
    return o;
  const double *p = a_c.data_ptr<double>();
  switch (dtype)
  {
  case c10::ScalarType::Half:
    narrow_run<uint16_t, 5, 10>(p, static_cast<uint16_t *>(o.data_ptr()), size);
    break;
  case c10::ScalarType::BFloat16:
    narrow_run<uint16_t, 8, 7>(p, static_cast<uint16_t *>(o.data_ptr()), size);
    break;
  default: // float32, the one dtype left after narrow_float64_tensors' check
    narrow_run<uint32_t, 8, 23>(p, static_cast<uint32_t *>(o.data_ptr()), size);
    break;
  }
  return o;
}
