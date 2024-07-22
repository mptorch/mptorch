#include "quant.h"
#include <torch/torch.h>
#include <tuple>
#include "binary8_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic,
            "Fixed Point Number Stochastic Quantization (CUDA)");
      m.def("fixed_point_quantize_stochastic_mask",
            &fixed_point_quantize_stochastic_mask,
            "Fixed Point Number Stochastic Quantization (CUDA)");
      m.def("block_quantize_stochastic", &block_quantize_stochastic,
            "Block Floating Point Number Stochastic Quantization (CUDA)");
      m.def("block_quantize_sim_stochastic", &block_quantize_sim_stochastic,
            "Block Floating Point Number Stochastic Quantization (CUDA)");
      m.def("float_quantize_stochastic", &float_quantize_stochastic,
            "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
      m.def("binary8_quantize_stochastic", &binary8_quantize_stochastic,
            "Low-Bitwidth P3109 Floating-Point Number Stochastic Quantization (CUDA)");
      m.def("binary8_quantize_truncate", &binary8_quantize_truncate,
            "Low-Bitwidth P3109 Floating-Point Number truncate Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest,
            "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask,
            "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("block_quantize_nearest", &block_quantize_nearest,
            "Block Floating Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("block_quantize_sim_nearest", &block_quantize_sim_nearest,
            "Block Floating Point Number Stochastic Quantization (CUDA)");
      m.def("float_quantize_nearest", &float_quantize_nearest,
            "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization "
            "(CUDA)");
      m.def("superfp_quantize_nearest", &superfp_quantize_nearest,
            "Low-Bitwidth Super Normal Floating Point Number Nearest Neighbor Quantization "
            "(CUDA)");
      m.def("binary8_quantize_nearest", &binary8_quantize_nearest,
            "Low-Bitwidth P3109 Floating-Point Number Nearest Quantization (CUDA)");

      py::enum_<OverflowPolicy>(m, "OverflowPolicy", py::arithmetic())
            .value("SATURATE_INFTY", OverflowPolicy::SATURATE_INFTY)
            .value("SATURATE_MAXFLOAT", OverflowPolicy::SATURATE_MAXFLOAT)
            .value("SATURATE_MAXFLOAT2", OverflowPolicy::SATURATE_MAXFLOAT2);
            
      m.def("float_quantize_nearest_mm", &float_quantize_nearest_mm,
            "Low-Bitwidth Floating Point Number GEMM Quantization (CUDA)");          
      m.def("float_quantize_nearest_bmm", &float_quantize_nearest_bmm,
            "Low-Bitwidth Floating Point Number BGEMM Quantization (CUDA)");
      m.def(
          "float_quantize_nearest_mm_fma", &float_quantize_nearest_mm_fma,
          "Low-Bitwidth Floating Point Number FMA-based GEMM Quantization (CUDA)");
      m.def(
          "float_quantize_nearest_bmm_fma", &float_quantize_nearest_bmm_fma,
          "Low-Bitwidth Floating Point Number FMA-based BGEMM Quantization (CUDA)");
      m.def("superfp_quantize_nearest_mm", &superfp_quantize_nearest_mm,
            "Low-Bitwidth SuperNormal Floating Point Number GEMM Quantization (CUDA)");
      m.def("superfp_quantize_nearest_bmm", &superfp_quantize_nearest_bmm,
            "Low-Bitwidth SuperNormal Floating Point Number BGEMM Quantization (CUDA)");
      m.def(
          "superfp_quantize_nearest_mm_fma", &superfp_quantize_nearest_mm_fma,
          "Low-Bitwidth SuperNormal Floating Point Number FMA-based GEMM Quantization (CUDA)");
      m.def(
          "superfp_quantize_nearest_bmm_fma", &superfp_quantize_nearest_bmm_fma,
          "Low-Bitwidth SuperNormal Floating Point Number FMA-based BGEMM Quantization (CUDA)");
      m.def("float_quantize_stochastic_mm", &float_quantize_stochastic_mm,
            "Low-Bitwidth Floating Point Number GEMM with Stochastic Quantization "
            "(CUDA)");
      m.def("float_quantize_stochastic_bmm", &float_quantize_stochastic_bmm,
            "Low-Bitwidth Floating Point Number BGEMM with Stochastic Quantization "
            "(CUDA)");
      m.def("float_quantize_stochastic_mm_fma", &float_quantize_stochastic_mm_fma,
            "Low-Bitwidth Floating Point Number FMA-based GEMM with Stochastic "
            "Quantization (CUDA)");
      m.def("float_quantize_stochastic_bmm_fma", &float_quantize_stochastic_bmm_fma,
            "Low-Bitwidth Floating Point Number FMA-based BGEMM with Stochastic "
            "Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest_mm", &fixed_point_quantize_nearest_mm,
            "Low-Bitwidth Fixed Point Number GEMM Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest_mm_fma",
            &fixed_point_quantize_nearest_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM Quantization (CUDA)");
      m.def("fixed_point_quantize_stochastic_mm",
            &fixed_point_quantize_stochastic_mm,
            "Low-Bitwidth Fixed Point Number GEMM with Stochastic Quantization "
            "(CUDA)");
      m.def("fixed_point_quantize_stochastic_mm_fma",
            &fixed_point_quantize_stochastic_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM with Stochastic "
            "Quantization (CUDA)");
}