// narrow_float64 on the CPU: rounds a float64 tensor once onto float32,
// float16 or bfloat16, with common/narrow_binary64.h's integer rounding of
// the word applied to every element on ATen's thread pool, as the
// elementwise quantizers are (utils.h). This exists because torch's own
// float64 to float16/bfloat16 conversion goes through float32 on both
// backends and so rounds twice; one rounding from the binary64 word is what
// a result computed in binary64 for a narrower tensor needs.

#include "../common/narrow_binary64.h"
#include "../common/narrow_host.h"
#include "../quant_ops.h"
#include "utils.h"
#include <cstring>

using at::Tensor;

namespace
{

  // o[i] = narrow(a[i]) for i in [0, size), where Word is the target
  // dtype's storage word (uint16_t or uint32_t) and ExpBits/ManBits its
  // field widths. The double is read as its 64-bit word through memcpy,
  // which the compiler turns into a plain load, since the rounding is
  // integer arithmetic on the word.
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

// The CPU kernel behind mptorch::narrow_float64: a new tensor of a's shape
// in `dtype` (float32, float16 or bfloat16), each element rounded once to
// nearest even from the float64 value. narrow_float64_tensors checks the
// dtypes and makes the input contiguous.
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
