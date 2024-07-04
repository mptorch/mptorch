#include "quant.h"
#include <torch/torch.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize_stochastic_mask",
        &fixed_point_quantize_stochastic_mask,
        "Fixed Point Number Stochastic Quantization with Mask (CPU)");
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic,
        "Fixed Point Number Stochastic Quantization (CPU)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic,
        "Block Floating Point Number Stochastic Quantization (CPU)");
  m.def("float_quantize_stochastic", &float_quantize_stochastic,
        "Low-Bitwidth Floating Point Number Stochastic Quantization (CPU)");
  m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask,
        "Fixed Point Number Nearest Quantization with Mask (CPU)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest,
        "Fixed Point Number Nearest Neighbor Quantization (CPU)");
  m.def("block_quantize_nearest", &block_quantize_nearest,
        "Block Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def(
      "float_quantize_nearest", &float_quantize_nearest,
      "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def(
      "superfp_quantize_nearest", &superfp_quantize_nearest,
      "Low-Bitwidth SuperNormal Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def("float_quantize_nearest_mm", &float_quantize_nearest_mm,
        "Low-Bitwidth GEMM (CPU)");
  m.def("float_quantize_nearest_mm_fma", &float_quantize_nearest_mm_fma,
        "Low-Bitwidth GEMM (CPU)");
  m.def("float_quantize_nearest_softmax_forward", &float_quantize_nearest_softmax_forward,
        "Low-Bitwidth Floating Point Softmax Forward using division. (CPU)");
  m.def("float_quantize_nearest_softmax_lse_forward", &float_quantize_nearest_softmax_lse_forward,
        "Low-Bitwidth Floating Point Softmax Forward using LogSumExp. (CPU)");
  m.def("float_quantize_nearest_softmax_backward", &float_quantize_nearest_softmax_backward,
        "Low-Bitwidth Floating Point Softmax Backward. (CPU)");
}