// The CPU dispatch-key registrations: binds each op declared in quant_ops.cpp
// to its CPU implementation (declared in quant_ops.h, defined in the kernel
// files of this directory). The op schema is declared once, in
// quant_ops.cpp's TORCH_LIBRARY block, and each backend registers its
// implementation under its own key in a block like this one, so a call to
// torch.ops.mptorch.<op> reaches the CPU or CUDA function according to the
// device of its tensors. cuda/cuda_ops.cu is this file's CUDA twin, and
// autograd_ops.cpp registers the Autograd key.

#include "../quant_ops.h"
#include <torch/library.h>

TORCH_LIBRARY_IMPL(mptorch, CPU, m)
{
  m.impl("binaryK_quant", TORCH_FN(binaryK_quantize_cpu));
  m.impl("superfp_quant", TORCH_FN(superfp_quantize_cpu));
  m.impl("narrow_float64", TORCH_FN(narrow_float64_cpu));
  m.impl("custom_matmul_binaryK", TORCH_FN(binaryK_matmul_cpu));
  m.impl("custom_matmul_superfp", TORCH_FN(superfp_matmul_cpu));
  m.impl("custom_matmul_binaryK_fma", TORCH_FN(binaryK_matmul_fma_cpu));
  m.impl("custom_matmul_superfp_fma", TORCH_FN(superfp_matmul_fma_cpu));
  m.impl("custom_matmul_binaryK_mixed", TORCH_FN(binaryK_matmul_mixed_cpu));
  m.impl("custom_matmul_superfp_mixed", TORCH_FN(superfp_matmul_mixed_cpu));
  m.impl("custom_matmul_binaryK_fma_mixed", TORCH_FN(binaryK_matmul_fma_mixed_cpu));
  m.impl("custom_matmul_superfp_fma_mixed", TORCH_FN(superfp_matmul_fma_mixed_cpu));
}
