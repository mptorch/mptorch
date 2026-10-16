// The MPS dispatch-key registrations: binds each op declared in quant_ops.cpp
// to its Apple GPU implementation, the twin of cpu/cpu_ops.cpp and
// cuda/cuda_ops.cpp. Compiled only where setup.py builds the MPS backend
// (macOS, with a torch built with MPS).
//
// Every op runs as Metal kernels whose results are the CPU backend's bit for
// bit, stochastic rounding included, up to what gemm.metal documents about
// the GPU's flushing of binary32 subnormals: the quantizers and their
// in-place twins (quantize.cpp), the GEMMs (custom_matmul_entry.cpp), their
// KAHAN, BLOCK and TREE twins (custom_matmul_accumulated_entry.cpp), the
// cross-family ones (custom_matmul_cross_entry.cpp), the conv ops
// (custom_conv_entry.cpp), the block GEMM (custom_matmul_block_entry.cpp)
// and the elementwise block ops (block_entry.cpp). The one exception is
// narrow_float64, whose input is a float64 tensor, which MPS cannot hold: it
// keeps torch's boxed CPU fallback, so that a call on an MPS tensor reaches
// the op's own dtype check and fails with its message rather than the
// dispatcher's, which names nothing a caller can act on.

#include "../quant_ops.h"
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

namespace
{
  void mps_via_cpu(const c10::OperatorHandle &op, torch::jit::Stack *stack)
  {
    at::native::cpu_fallback(op, stack);
  }
} // namespace

TORCH_LIBRARY_IMPL(mptorch, MPS, m)
{
  m.impl("binaryK_quant", TORCH_FN(binaryK_quantize_mps));
  m.impl("superfp_quant", TORCH_FN(superfp_quantize_mps));
  m.impl("binaryK_quant_", TORCH_FN(binaryK_quantize_mps_));
  m.impl("superfp_quant_", TORCH_FN(superfp_quantize_mps_));
  m.impl("narrow_float64", torch::CppFunction::makeFromBoxedFunction<&mps_via_cpu>());
  m.impl("custom_matmul_binaryK", TORCH_FN(binaryK_matmul_mps));
  m.impl("custom_matmul_superfp", TORCH_FN(superfp_matmul_mps));
  m.impl("custom_matmul_binaryK_fma", TORCH_FN(binaryK_matmul_fma_mps));
  m.impl("custom_matmul_superfp_fma", TORCH_FN(superfp_matmul_fma_mps));
  m.impl("custom_matmul_binaryK_mixed", TORCH_FN(binaryK_matmul_mixed_mps));
  m.impl("custom_matmul_superfp_mixed", TORCH_FN(superfp_matmul_mixed_mps));
  m.impl("custom_matmul_binaryK_fma_mixed", TORCH_FN(binaryK_matmul_fma_mixed_mps));
  m.impl("custom_matmul_superfp_fma_mixed", TORCH_FN(superfp_matmul_fma_mixed_mps));
  m.impl("custom_matmul_binaryK_accumulated", TORCH_FN(binaryK_matmul_accumulated_mps));
  m.impl("custom_matmul_superfp_accumulated", TORCH_FN(superfp_matmul_accumulated_mps));
  m.impl("custom_matmul_binaryK_fma_accumulated", TORCH_FN(binaryK_matmul_fma_accumulated_mps));
  m.impl("custom_matmul_superfp_fma_accumulated", TORCH_FN(superfp_matmul_fma_accumulated_mps));
  m.impl("custom_conv_binaryK", TORCH_FN(binaryK_conv_mps));
  m.impl("custom_conv_binaryK_mixed", TORCH_FN(binaryK_conv_mixed_mps));
  m.impl("custom_conv_binaryK_fma", TORCH_FN(binaryK_conv_fma_mps));
  m.impl("custom_conv_binaryK_fma_mixed", TORCH_FN(binaryK_conv_fma_mixed_mps));
  m.impl("custom_conv_superfp", TORCH_FN(superfp_conv_mps));
  m.impl("custom_conv_superfp_mixed", TORCH_FN(superfp_conv_mixed_mps));
  m.impl("custom_conv_superfp_fma", TORCH_FN(superfp_conv_fma_mps));
  m.impl("custom_conv_superfp_fma_mixed", TORCH_FN(superfp_conv_fma_mixed_mps));
  m.impl("block_pack", TORCH_FN(block_pack_mps));
  m.impl("block_unpack", TORCH_FN(block_unpack_mps));
  m.impl("block_quant", TORCH_FN(block_quantize_mps));
  m.impl("block_quant_", TORCH_FN(block_quantize_mps_));
  m.impl("custom_matmul_block", TORCH_FN(block_matmul_mps));
  m.impl("custom_matmul_binaryK_superfp", TORCH_FN(binaryK_superfp_matmul_mps));
  m.impl("custom_matmul_superfp_binaryK", TORCH_FN(superfp_binaryK_matmul_mps));
  m.impl("custom_matmul_binaryK_superfp_accumulated", TORCH_FN(binaryK_superfp_matmul_accumulated_mps));
  m.impl("custom_matmul_superfp_binaryK_accumulated", TORCH_FN(superfp_binaryK_matmul_accumulated_mps));
  m.impl("custom_conv_binaryK_superfp", TORCH_FN(binaryK_superfp_conv_mps));
  m.impl("custom_conv_superfp_binaryK", TORCH_FN(superfp_binaryK_conv_mps));
}
