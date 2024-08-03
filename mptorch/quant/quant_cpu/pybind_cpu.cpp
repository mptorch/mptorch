#include "quant.h"
#include "binary8.h"
#include <torch/torch.h>
#include <tuple>

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
      TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
      CHECK_CPU(x);   \
      CHECK_CONTIGUOUS(x)

Tensor binary8_quantize_stochastic(Tensor a, int P, int prng_bits, bool is_signed, OverflowPolicy overflow_policy, bool subnormals)
{
      CHECK_INPUT(a);
      return binary8_quantize_stochastic_cpu(a, P, prng_bits, is_signed, overflow_policy, subnormals);
}

Tensor binary8_quantize_truncate(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals)
{
      CHECK_INPUT(a);
      return binary8_quantize_truncate_cpu(a, P, is_signed, overflow_policy, subnormals);
}

Tensor binary8_quantize_nearest(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals)
{
      CHECK_INPUT(a);
      return binary8_quantize_nearest_cpu(a, P, is_signed, overflow_policy, subnormals);
}



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
      m.def("float_quantize_nearest", &float_quantize_nearest,
            "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CPU)");
      m.def("superfp_quantize_nearest", &superfp_quantize_nearest,
            "Low-Bitwidth SuperNormal Floating Point Number Nearest Neighbor Quantization (CPU)");
      m.def("float_quantize_nearest_mm", &float_quantize_nearest_mm,
            "Low-Bitwidth GEMM (CPU)");
      m.def("float_quantize_nearest_mm_fma", &float_quantize_nearest_mm_fma,
            "Low-Bitwidth GEMM (CPU)");
      
      m.def("binary8_quantize_stochastic", &binary8_quantize_stochastic,
            "Low-Bitwidth P3109 Floating-Point Number Stochastic Quantization (CPU)");
      m.def("binary8_quantize_truncate", &binary8_quantize_truncate,
            "Low-Bitwidth P3109 Floating-Point Number truncate Quantization (CPU)");
      m.def("binary8_quantize_nearest", &binary8_quantize_nearest,
            "Low-Bitwidth P3109 Floating-Point Number Nearest Quantization (CPU)");

      py::enum_<OverflowPolicy>(m, "OverflowPolicy", py::arithmetic(), py::module_local())
            .value("SATURATE_INFTY", OverflowPolicy::SATURATE_INFTY)
            .value("SATURATE_MAXFLOAT", OverflowPolicy::SATURATE_MAXFLOAT)
            .value("SATURATE_MAXFLOAT2", OverflowPolicy::SATURATE_MAXFLOAT2);
      
      m.def("float_quantize_nearest_softmax_forward", &float_quantize_nearest_softmax_forward,
            "Low-Bitwidth Floating Point Softmax Forward using division. (CPU)");
      m.def("float_quantize_nearest_softmax_lse_forward", &float_quantize_nearest_softmax_lse_forward,
            "Low-Bitwidth Floating Point Softmax Forward using LogSumExp. (CPU)");
      m.def("float_quantize_nearest_softmax_backward", &float_quantize_nearest_softmax_backward,
            "Low-Bitwidth Floating Point Softmax Backward. (CPU)");
      
      m.def("superfp_quantize_nearest_softmax_forward", &superfp_quantize_nearest_softmax_forward,
            "Low-Bitwidth Super Floating Point Softmax Forward using division. (CPU)");
      m.def("superfp_quantize_nearest_softmax_lse_forward", &superfp_quantize_nearest_softmax_lse_forward,
            "Low-Bitwidth Super Floating Point Softmax Forward using LogSumExp. (CPU)");
      m.def("superfp_quantize_nearest_softmax_backward", &superfp_quantize_nearest_softmax_backward,
            "Low-Bitwidth Super Floating Point Softmax Backward. (CPU)");
      
      m.def("binary8_quantize_nearest_softmax_forward", &binary8_quantize_nearest_softmax_forward,
            "Binary8 Softmax Forward using division. (CPU)");
      m.def("binary8_quantize_nearest_softmax_lse_forward", &binary8_quantize_nearest_softmax_lse_forward,
            "Binary8 Softmax Forward using LogSumExp. (CPU)");
      m.def("binary8_quantize_nearest_softmax_backward", &binary8_quantize_nearest_softmax_backward,
            "Binary8 Softmax Backward. (CPU)");
}