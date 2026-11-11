// Binds the CUDA implementations to the op schemas declared in
// ../quant_ops.cpp. The functions are declared in ../quant_ops.h and defined
// in this directory: the three elementwise quantizers and their in-place twins
// in binaryK_kernel.cu, superfp_kernel.cu and fixedpoint_kernel.cu, the
// narrowing op in narrow_entry.cpp, and the GEMM, conv, block and
// cross-family ops in the *_entry.cpp files.
//
// Every op goes through OnOperandDevice (device_guard.h), which runs it on
// its first CUDA tensor's device: register a new one with MPT_CUDA_IMPL too.
#include "../quant_ops.h"
#include "device_guard.h"
#include <torch/library.h>

#define MPT_CUDA_IMPL(name, fn) m.impl(name, TORCH_FN(mptorch::OnOperandDevice<&fn>::call))

TORCH_LIBRARY_IMPL(mptorch, CUDA, m)
{
    MPT_CUDA_IMPL("binaryK_quant", binaryK_quantize_cuda);
    MPT_CUDA_IMPL("fixedpoint_quant", fixedpoint_quantize_cuda);
    MPT_CUDA_IMPL("superfp_quant", superfp_quantize_cuda);
    MPT_CUDA_IMPL("binaryK_quant_", binaryK_quantize_cuda_);
    MPT_CUDA_IMPL("fixedpoint_quant_", fixedpoint_quantize_cuda_);
    MPT_CUDA_IMPL("superfp_quant_", superfp_quantize_cuda_);
    MPT_CUDA_IMPL("narrow_float64", narrow_float64_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK", binaryK_matmul_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp", superfp_matmul_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_fma", binaryK_matmul_fma_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_fma", superfp_matmul_fma_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_mixed", binaryK_matmul_mixed_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_mixed", superfp_matmul_mixed_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_fma_mixed", binaryK_matmul_fma_mixed_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_fma_mixed", superfp_matmul_fma_mixed_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_accumulated", binaryK_matmul_accumulated_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_accumulated", superfp_matmul_accumulated_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_fma_accumulated", binaryK_matmul_fma_accumulated_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_fma_accumulated", superfp_matmul_fma_accumulated_cuda);
    MPT_CUDA_IMPL("custom_conv_binaryK", binaryK_conv_cuda);
    MPT_CUDA_IMPL("custom_conv_binaryK_mixed", binaryK_conv_mixed_cuda);
    MPT_CUDA_IMPL("custom_conv_binaryK_fma", binaryK_conv_fma_cuda);
    MPT_CUDA_IMPL("custom_conv_binaryK_fma_mixed", binaryK_conv_fma_mixed_cuda);
    MPT_CUDA_IMPL("custom_conv_superfp", superfp_conv_cuda);
    MPT_CUDA_IMPL("custom_conv_superfp_mixed", superfp_conv_mixed_cuda);
    MPT_CUDA_IMPL("custom_conv_superfp_fma", superfp_conv_fma_cuda);
    MPT_CUDA_IMPL("custom_conv_superfp_fma_mixed", superfp_conv_fma_mixed_cuda);
    MPT_CUDA_IMPL("block_pack", block_pack_cuda);
    MPT_CUDA_IMPL("block_unpack", block_unpack_cuda);
    MPT_CUDA_IMPL("block_quant", block_quantize_cuda);
    MPT_CUDA_IMPL("block_quant_", block_quantize_cuda_);
    MPT_CUDA_IMPL("custom_matmul_block", block_matmul_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_superfp", binaryK_superfp_matmul_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_binaryK", superfp_binaryK_matmul_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_superfp_accumulated", binaryK_superfp_matmul_accumulated_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_binaryK_accumulated", superfp_binaryK_matmul_accumulated_cuda);
    MPT_CUDA_IMPL("custom_conv_binaryK_superfp", binaryK_superfp_conv_cuda);
    MPT_CUDA_IMPL("custom_conv_superfp_binaryK", superfp_binaryK_conv_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_odd", binaryK_matmul_odd_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_odd", superfp_matmul_odd_cuda);
    MPT_CUDA_IMPL("custom_matmul_binaryK_fma_odd", binaryK_matmul_fma_odd_cuda);
    MPT_CUDA_IMPL("custom_matmul_superfp_fma_odd", superfp_matmul_fma_odd_cuda);
    MPT_CUDA_IMPL("_carrier_odd", carrier_odd_cuda);
    MPT_CUDA_IMPL("_stochastic_cast", stochastic_cast_cuda);
}
