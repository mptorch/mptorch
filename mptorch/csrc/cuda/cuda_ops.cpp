// Binds the CUDA implementations to the op schemas declared in
// ../quant_ops.cpp. The functions are declared in ../quant_ops.h and defined
// in this directory: the two elementwise quantizers and their in-place twins
// in binaryK_kernel.cu and superfp_kernel.cu, the narrowing op in narrow_entry.cpp, and the eight
// GEMM ops in custom_matmul_entry.cpp.
#include "../quant_ops.h"
#include <torch/library.h>

TORCH_LIBRARY_IMPL(mptorch, CUDA, m)
{
    m.impl("binaryK_quant", TORCH_FN(binaryK_quantize_cuda));
    m.impl("superfp_quant", TORCH_FN(superfp_quantize_cuda));
    m.impl("binaryK_quant_", TORCH_FN(binaryK_quantize_cuda_));
    m.impl("superfp_quant_", TORCH_FN(superfp_quantize_cuda_));
    m.impl("narrow_float64", TORCH_FN(narrow_float64_cuda));
    m.impl("custom_matmul_binaryK", TORCH_FN(binaryK_matmul_cuda));
    m.impl("custom_matmul_superfp", TORCH_FN(superfp_matmul_cuda));
    m.impl("custom_matmul_binaryK_fma", TORCH_FN(binaryK_matmul_fma_cuda));
    m.impl("custom_matmul_superfp_fma", TORCH_FN(superfp_matmul_fma_cuda));
    m.impl("custom_matmul_binaryK_mixed", TORCH_FN(binaryK_matmul_mixed_cuda));
    m.impl("custom_matmul_superfp_mixed", TORCH_FN(superfp_matmul_mixed_cuda));
    m.impl("custom_matmul_binaryK_fma_mixed", TORCH_FN(binaryK_matmul_fma_mixed_cuda));
    m.impl("custom_matmul_superfp_fma_mixed", TORCH_FN(superfp_matmul_fma_mixed_cuda));
}
