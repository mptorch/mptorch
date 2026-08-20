#include "../quant_ops.h"
#include <torch/library.h>

TORCH_LIBRARY_IMPL(mptorch, CUDA, m)
{
    m.impl("binaryK_quant", TORCH_FN(binaryK_quantize_cuda));
    m.impl("superfp_quant", TORCH_FN(superfp_quantize_cuda));
}
