#include "quant.h"
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("float_quantize_nearest_even", &float_quantize_nearest_even,
            "Custom-Precision IEEE-754-like Floating-Point Quantization with "
            "Nearest Rounding Ties To Even (CPU)");
      m.def("float_quantize_nearest_away", &float_quantize_nearest_away,
            "Custom-Precision IEEE-754-like Floating-Point Quantization with "
            "Nearest Rounding Ties To Away (CPU)");
      m.def("float_quantize_up", &float_quantize_up,
            "Custom-Precision IEEE-754-like Floating-Point Quantization with "
            "Rounding Towards Positive (CPU)");
      m.def("float_quantize_down", &float_quantize_down,
            "Custom-Precision IEEE-754-like Floating-Point Quantization with "
            "Rounding Towards Negative (CPU)");
      m.def("float_quantize_zero", &float_quantize_zero,
            "Custom-Precision IEEE-754-like Floating-Point Quantization with "
            "Rounding Towards Zero (CPU)");
      m.def("float_quantize_stochastic", &float_quantize_stochastic,
            "Custom-Precision IEEE-754-like Floating-Point Quantization with "
            "Stochastic Rounding (CPU)");

      m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest,
            "Fixed Point Number Nearest Neighbor Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask,
            "Fixed Point Number Nearest Quantization with Mask (CPU)");
      m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic,
            "Fixed Point Number Stochastic Quantization (CPU)");
      m.def("fixed_point_quantize_stochastic_mask",
            &fixed_point_quantize_stochastic_mask,
            "Fixed Point Number Stochastic Quantization with Mask (CPU)");

      m.def("block_quantize_nearest", &block_quantize_nearest,
            "Block Floating Point Number Nearest Neighbor Quantization (CPU)");
      m.def("block_quantize_stochastic", &block_quantize_stochastic,
            "Block Floating Point Number Stochastic Quantization (CPU)");

      m.def("superfp_quantize_nearest", &superfp_quantize_nearest,
            "Low-Bitwidth SuperNormal Floating Point Number Nearest Neighbor "
            "Quantization (CPU)");

      m.def("fp_quantize", &fp_quantize,
            "Custom-precision IEEE-754-like Floating-Point Quantization (CPU)");

      m.def("binaryK_quantize", &binaryK_quantize,
            "Custom-precision P3109 Floating-Point Quantization (CPU)");

      m.def("superfp_quantize", &superfp_quantize,
            "Custom-precision SuperNormal Floating-Point Quantization (CPU)");

      py::enum_<SaturationMode>(m, "SaturationMode", py::arithmetic(),
                                py::module_local())
          .value("SAT_FINITE", SaturationMode::SAT_FINITE)
          .value("SAT_PROPAGATE", SaturationMode::SAT_PROPAGATE)
          .value("OVF_INF", SaturationMode::OVF_INF);

      py::enum_<SubnormalsMode>(m, "SubnormalsMode", py::arithmetic(),
                                py::module_local())
          .value("SUBNORMALS", SubnormalsMode::SUBNORMALS)
          .value("NORMALS", SubnormalsMode::NORMALS)
          .value("EXTENDED_NORMALS", SubnormalsMode::EXTENDED_NORMALS);

      py::enum_<RoundMode>(m, "RoundMode", py::arithmetic(),
                           py::module_local())
          .value("RNE", RoundMode::RNE)
          .value("RNA", RoundMode::RNA)
          .value("RU", RoundMode::RU)
          .value("RD", RoundMode::RD)
          .value("RZ", RoundMode::RZ)
          .value("SR", RoundMode::SR);

      m.def("float_quantize_nearest_mm", &float_quantize_nearest_mm,
            "Low-Bitwidth Floating Point Number GEMM Quantization (CPU)");
      m.def("float_quantize_nearest_bmm", &float_quantize_nearest_bmm,
            "Low-Bitwidth Floating Point Number BGEMM Quantization (CPU)");
      m.def("float_quantize_nearest_mm_fma", &float_quantize_nearest_mm_fma,
            "Low-Bitwidth Floating Point Number FMA-based GEMM Quantization (CPU)");
      m.def(
          "float_quantize_nearest_bmm_fma", &float_quantize_nearest_bmm_fma,
          "Low-Bitwidth Floating Point Number FMA-based BGEMM Quantization (CPU)");

      m.def("float_quantize_stochastic_mm", &float_quantize_stochastic_mm,
            "Low-Bitwidth Floating Point Number GEMM with Stochastic Quantization "
            "(CPU)");
      m.def("float_quantize_stochastic_bmm", &float_quantize_stochastic_bmm,
            "Low-Bitwidth Floating Point Number BGEMM with Stochastic Quantization "
            "(CPU)");
      m.def("float_quantize_stochastic_mm_fma", &float_quantize_stochastic_mm_fma,
            "Low-Bitwidth Floating Point Number FMA-based GEMM with Stochastic "
            "Quantization (CPU)");
      m.def("float_quantize_stochastic_bmm_fma", &float_quantize_stochastic_bmm_fma,
            "Low-Bitwidth Floating Point Number FMA-based BGEMM with Stochastic "
            "Quantization (CPU)");

      m.def(
          "superfp_quantize_nearest_mm", &superfp_quantize_nearest_mm,
          "Low-Bitwidth SuperNormal Floating Point Number GEMM Quantization (CPU)");
      m.def("superfp_quantize_nearest_bmm", &superfp_quantize_nearest_bmm,
            "Low-Bitwidth SuperNormal Floating Point Number BGEMM Quantization "
            "(CPU)");
      m.def("superfp_quantize_nearest_mm_fma", &superfp_quantize_nearest_mm_fma,
            "Low-Bitwidth SuperNormal Floating Point Number FMA-based GEMM "
            "Quantization (CPU)");
      m.def("superfp_quantize_nearest_bmm_fma", &superfp_quantize_nearest_bmm_fma,
            "Low-Bitwidth SuperNormal Floating Point Number FMA-based BGEMM "
            "Quantization (CPU)");

      m.def("fixed_point_quantize_nearest_mm", &fixed_point_quantize_nearest_mm,
            "Low-Bitwidth Fixed Point Number GEMM Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_bmm", &fixed_point_quantize_nearest_bmm,
            "Low-Bitwidth Fixed Point Number BGEMM Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_mm_fma",
            &fixed_point_quantize_nearest_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM Quantization (CPU)");
      m.def("fixed_point_quantize_nearest_mm_fma",
            &fixed_point_quantize_nearest_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based BGEMM Quantization (CPU)");

      m.def("fixed_point_quantize_stochastic_mm",
            &fixed_point_quantize_stochastic_mm,
            "Low-Bitwidth Fixed Point Number GEMM with Stochastic Quantization "
            "(CPU)");
      m.def("fixed_point_quantize_stochastic_bmm",
            &fixed_point_quantize_stochastic_bmm,
            "Low-Bitwidth Fixed Point Number BGEMM with Stochastic Quantization "
            "(CPU)");
      m.def("fixed_point_quantize_stochastic_mm_fma",
            &fixed_point_quantize_stochastic_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM with Stochastic "
            "Quantization (CPU)");
      m.def("fixed_point_quantize_stochastic_bmm_fma",
            &fixed_point_quantize_stochastic_bmm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM with Stochastic "
            "Quantization (CPU)");

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
            "Low-Bitwidth Super Floating Point Softmax Forward using division. "
            "(CPU)");
      m.def("superfp_quantize_nearest_softmax_lse_forward",
            &superfp_quantize_nearest_softmax_lse_forward,
            "Low-Bitwidth Super Floating Point Softmax Forward using LogSumExp. "
            "(CPU)");
      m.def("superfp_quantize_nearest_softmax_backward",
            &superfp_quantize_nearest_softmax_backward,
            "Low-Bitwidth Super Floating Point Softmax Backward. (CPU)");

      m.def("float_quantize_layernorm_forward", &float_quantize_layernorm_forward,
            "Low-Bitwidth Floating Point Layer Normalization (CPU)");
      m.def("float_quantize_layernorm_backward", &float_quantize_layernorm_backward,
            "Low-Bitwidth Floating Point Layer Normalization Backward (CPU)");

      m.def("superfp_quantize_layernorm_forward",
            &superfp_quantize_layernorm_forward,
            "Low-Bitwidth Super Floating Point Layer Normalization (CPU)");
      m.def("superfp_quantize_layernorm_backward",
            &superfp_quantize_layernorm_backward,
            "Low-Bitwidth Super Floating Point Layer Normalization Backward (CPU)");
}