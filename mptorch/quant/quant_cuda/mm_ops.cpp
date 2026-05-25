#include "../mm_ops.h"
#include <torch/library.h>

TORCH_LIBRARY_IMPL(mptorch, CUDA, m)
{
  m.impl("fp_mm", TORCH_FN(fp_mm_cuda));
  m.impl("superfp_mm", TORCH_FN(superfp_mm_cuda));
  m.impl("fxp_mm", TORCH_FN(fxp_mm_cuda));
}
