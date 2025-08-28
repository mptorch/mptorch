#include "quant.h"
#include "binary8.h"
#include <torch/torch.h>
#include <tuple>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("float_quantize_nearest",
            &float_quantize_nearest,
            "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CPU)");
      m.def("float_quantize_stochastic",
            &float_quantize_stochastic,
            "Low-Bitwidth Floating Point Number Stochastic Quantization (CPU)");

      m.def("fixed_point_quantize_nearest",
            &fixed_point_quantize_nearest,
            "Fixed Point Number Nearest Neighbor Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_mask",
            &fixed_point_quantize_nearest_mask,
            "Fixed Point Number Nearest Quantization with Mask (CPU)");
      m.def("fixed_point_quantize_stochastic",
            &fixed_point_quantize_stochastic,
            "Fixed Point Number Stochastic Quantization (CPU)");
      m.def("fixed_point_quantize_stochastic_mask",
            &fixed_point_quantize_stochastic_mask,
            "Fixed Point Number Stochastic Quantization with Mask (CPU)");

      m.def("block_quantize_nearest",
            &block_quantize_nearest,
            "Block Floating Point Number Nearest Neighbor Quantization (CPU)");
      m.def("block_quantize_stochastic",
            &block_quantize_stochastic,
            "Block Floating Point Number Stochastic Quantization (CPU)");

      m.def("superfp_quantize_nearest",
            &superfp_quantize_nearest,
            "Low-Bitwidth SuperNormal Floating Point Number Nearest Neighbor Quantization (CPU)");

      m.def("binary8_quantize_nearest",
            &binary8_quantize_nearest,
            "Low-Bitwidth P3109 Floating-Point Number Nearest Quantization (CPU)");
      m.def("binary8_quantize_truncate",
            &binary8_quantize_truncate,
            "Low-Bitwidth P3109 Floating-Point Number truncate Quantization (CPU)");
      m.def("binary8_quantize_stochastic",
            &binary8_quantize_stochastic,
            "Low-Bitwidth P3109 Floating-Point Number Stochastic Quantization (CPU)");

      py::enum_<OverflowPolicy>(m, "OverflowPolicy", py::arithmetic(), py::module_local())
          .value("SATURATE_INFTY", OverflowPolicy::SATURATE_INFTY)
          .value("SATURATE_MAXFLOAT", OverflowPolicy::SATURATE_MAXFLOAT)
          .value("SATURATE_MAXFLOAT2", OverflowPolicy::SATURATE_MAXFLOAT2);

      m.def("float_quantize_nearest_mm",
            &float_quantize_nearest_mm,
            "Low-Bitwidth Floating Point Number GEMM Quantization (CPU)");
      m.def("float_quantize_nearest_bmm",
            &float_quantize_nearest_bmm,
            "Low-Bitwidth Floating Point Number BGEMM Quantization (CPU)");
      m.def("float_quantize_nearest_mm_fma",
            &float_quantize_nearest_mm_fma,
            "Low-Bitwidth Floating Point Number FMA-based GEMM Quantization (CPU)");
      m.def("float_quantize_nearest_bmm_fma",
            &float_quantize_nearest_bmm_fma,
            "Low-Bitwidth Floating Point Number FMA-based BGEMM Quantization (CPU)");

      m.def("float_quantize_stochastic_mm",
            &float_quantize_stochastic_mm,
            "Low-Bitwidth Floating Point Number GEMM with Stochastic Quantization (CPU)");
      m.def("float_quantize_stochastic_bmm",
            &float_quantize_stochastic_bmm,
            "Low-Bitwidth Floating Point Number BGEMM with Stochastic Quantization (CPU)");
      m.def("float_quantize_stochastic_mm_fma",
            &float_quantize_stochastic_mm_fma,
            "Low-Bitwidth Floating Point Number FMA-based GEMM with Stochastic Quantization (CPU)");
      m.def("float_quantize_stochastic_bmm_fma",
            &float_quantize_stochastic_bmm_fma,
            "Low-Bitwidth Floating Point Number FMA-based BGEMM with Stochastic Quantization (CPU)");

      m.def("superfp_quantize_nearest_mm",
            &superfp_quantize_nearest_mm,
            "Low-Bitwidth SuperNormal Floating Point Number GEMM Quantization (CPU)");
      m.def("superfp_quantize_nearest_bmm",
            &superfp_quantize_nearest_bmm,
            "Low-Bitwidth SuperNormal Floating Point Number BGEMM Quantization (CPU)");
      m.def("superfp_quantize_nearest_mm_fma",
            &superfp_quantize_nearest_mm_fma,
            "Low-Bitwidth SuperNormal Floating Point Number FMA-based GEMM Quantization (CPU)");
      m.def("superfp_quantize_nearest_bmm_fma",
            &superfp_quantize_nearest_bmm_fma,
            "Low-Bitwidth SuperNormal Floating Point Number FMA-based BGEMM Quantization (CPU)");

      m.def("fixed_point_quantize_nearest_mm",
            &fixed_point_quantize_nearest_mm,
            "Low-Bitwidth Fixed Point Number GEMM Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_bmm",
            &fixed_point_quantize_nearest_bmm,
            "Low-Bitwidth Fixed Point Number BGEMM Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_mm_fma",
            &fixed_point_quantize_nearest_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_mm_fma",
            &fixed_point_quantize_nearest_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based BGEMM Quantization (CPU)");

      m.def("fixed_point_quantize_stochastic_mm",
            &fixed_point_quantize_stochastic_mm,
            "Low-Bitwidth Fixed Point Number GEMM with Stochastic Quantization (CPU)");
      m.def("fixed_point_quantize_stochastic_bmm",
            &fixed_point_quantize_stochastic_bmm,
            "Low-Bitwidth Fixed Point Number BGEMM with Stochastic Quantization (CPU)");
      m.def("fixed_point_quantize_stochastic_mm_fma",
            &fixed_point_quantize_stochastic_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM with Stochastic Quantization (CPU)");
      m.def("fixed_point_quantize_stochastic_bmm_fma",
            &fixed_point_quantize_stochastic_bmm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM with Stochastic Quantization (CPU)");

      m.def("float_quantize_nearest_softmax_forward",
            &float_quantize_nearest_softmax_forward,
            "Low-Bitwidth Floating Point Softmax Forward using division. (CPU)");
      m.def("float_quantize_nearest_softmax_lse_forward",
            &float_quantize_nearest_softmax_lse_forward,
            "Low-Bitwidth Floating Point Softmax Forward using LogSumExp. (CPU)");
      m.def("float_quantize_nearest_softmax_backward",
            &float_quantize_nearest_softmax_backward,
            "Low-Bitwidth Floating Point Softmax Backward. (CPU)");

      m.def("superfp_quantize_nearest_softmax_forward",
            &superfp_quantize_nearest_softmax_forward,
            "Low-Bitwidth Super Floating Point Softmax Forward using division. (CPU)");
      m.def("superfp_quantize_nearest_softmax_lse_forward",
            &superfp_quantize_nearest_softmax_lse_forward,
            "Low-Bitwidth Super Floating Point Softmax Forward using LogSumExp. (CPU)");
      m.def("superfp_quantize_nearest_softmax_backward",
            &superfp_quantize_nearest_softmax_backward,
            "Low-Bitwidth Super Floating Point Softmax Backward. (CPU)");

      m.def("binary8_quantize_nearest_softmax_forward",
            &binary8_quantize_nearest_softmax_forward,
            "Binary8 Softmax Forward using division. (CPU)");
      m.def("binary8_quantize_nearest_softmax_lse_forward",
            &binary8_quantize_nearest_softmax_lse_forward,
            "Binary8 Softmax Forward using LogSumExp. (CPU)");
      m.def("binary8_quantize_nearest_softmax_backward",
            &binary8_quantize_nearest_softmax_backward,
            "Binary8 Softmax Backward. (CPU)");

      m.def("float_quantize_layernorm_forward",
            &float_quantize_layernorm_forward,
            "Low-Bitwidth Floating Point Layer Normalization (CPU)");
      m.def("float_quantize_layernorm_backward",
            &float_quantize_layernorm_backward,
            "Low-Bitwidth Floating Point Layer Normalization Backward (CPU)");

      m.def("superfp_quantize_layernorm_forward",
            &superfp_quantize_layernorm_forward,
            "Low-Bitwidth Super Floating Point Layer Normalization (CPU)");
      m.def("superfp_quantize_layernorm_backward",
            &superfp_quantize_layernorm_backward,
            "Low-Bitwidth Super Floating Point Layer Normalization Backward (CPU)");

      m.def("binary8_quantize_layernorm_forward",
            &binary8_quantize_layernorm_forward,
            "Low-Bitwidth binary8 Layer Normalization (CPU)");
      m.def("binary8_quantize_layernorm_backward",
            &binary8_quantize_layernorm_backward,
            "Low-Bitwidth binary8 Layer Normalization Backward (CPU)");
}