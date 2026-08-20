#include "../quant_ops.h"
#include <torch/library.h>

TORCH_LIBRARY_IMPL(mptorch, CPU, m)
{
  m.impl("binaryK_quant", TORCH_FN(binaryK_quantize_cpu));
  m.impl("superfp_quant", TORCH_FN(superfp_quantize_cpu));
}
