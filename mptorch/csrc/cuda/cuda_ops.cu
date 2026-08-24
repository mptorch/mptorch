#include "../quant_ops.h"
#include <torch/library.h>

TORCH_LIBRARY_IMPL(mptorch, CUDA, m)
{
    m.impl("binaryK_quant", TORCH_FN(binaryK_quantize_cuda));
    m.impl("superfp_quant", TORCH_FN(superfp_quantize_cuda));
    m.impl("custom_matmul_binaryK", TORCH_FN(binaryK_matmul_cuda));
    m.impl("custom_matmul_superfp", TORCH_FN(superfp_matmul_cuda));
    m.impl("custom_matmul_binaryK_fma", TORCH_FN(binaryK_matmul_fma_cuda));
    m.impl("custom_matmul_superfp_fma", TORCH_FN(superfp_matmul_fma_cuda));
}
