#include "quant.h"
#include <torch/torch.h>
#include <tuple>

using namespace at;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
      TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
      CHECK_CUDA(x);   \
      CHECK_CONTIGUOUS(x)

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool use_clamp,
                                    bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_nearest_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl, bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_nearest_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_nearest(Tensor a, int wl, int dim)
{
      CHECK_INPUT(a);
      return block_quantize_nearest_cuda(a, wl, dim);
}

Tensor block_quantize_sim_nearest(Tensor a, int wl)
{
      CHECK_INPUT(a);
      return block_quantize_sim_nearest_cuda(a, wl);
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits,
                              bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      return float_quantize_nearest_cuda(a, man_bits, exp_bits, subnormals,
                                         saturate);
}

Tensor superfp_quantize_nearest(Tensor a, int man_bits, int exp_bits, 
                              bool saturate)
{
      CHECK_INPUT(a);
      return superfp_quantize_nearest_cuda(a, man_bits, exp_bits, saturate);
}

Tensor p3109_quantize_nearest(Tensor a, int P, bool is_signed, bool subnormals)
{
      CHECK_INPUT(a);
      return p3109_quantize_nearest_cuda(a, P, is_signed, subnormals);
}

Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool use_clamp,
                                       bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_stochastic_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl, bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_stochastic_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_stochastic(Tensor a, int wl, int dim)
{
      CHECK_INPUT(a);
      return block_quantize_stochastic_cuda(a, wl, dim);
}

Tensor block_quantize_sim_stochastic(Tensor a, int wl)
{
      CHECK_INPUT(a);
      return block_quantize_sim_stochastic_cuda(a, wl);
}

Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits,
                                 bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      return float_quantize_stochastic_cuda(a, man_bits, exp_bits, subnormals,
                                            saturate);
}

Tensor p3109_quantize_stochastic(Tensor a, int P, int prng_bits, bool is_signed, bool subnormals)
{
      CHECK_INPUT(a);
      return p3109_quantize_stochastic_cuda(a, P, prng_bits, is_signed, subnormals);
}

void float_quantize_nearest_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                               int K, int man_mul, int exp_mul, int man_add,
                               int exp_add, bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_mm_cuda(a, b, c, M, N, K, man_mul, exp_mul, man_add,
                                     exp_add, subnormals, saturate);
      return;
}

void float_quantize_nearest_bmm(Tensor a, Tensor b, Tensor c, int M, int N,
                                int K, int man_mul, int exp_mul, int man_add,
                                int exp_add, bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_bmm_cuda(a, b, c, M, N, K, man_mul, exp_mul, man_add,
                                      exp_add, subnormals, saturate);
      return;
}

void float_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c, int M, int N,
                                   int K, int man_fma, int exp_fma,
                                   bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                         subnormals, saturate);
      return;
}

void float_quantize_nearest_bmm_fma(Tensor a, Tensor b, Tensor c, int M, int N,
                                    int K, int man_fma, int exp_fma,
                                    bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_bmm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                          subnormals, saturate);
      return;
}

void superfp_quantize_nearest_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                               int K, int man_mul, int exp_mul, int man_add,
                               int exp_add, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_mm_cuda(a, b, c, M, N, K, man_mul, exp_mul, man_add,
                                     exp_add, saturate);
      return;
}

void superfp_quantize_nearest_bmm(Tensor a, Tensor b, Tensor c, int M, int N,
                                int K, int man_mul, int exp_mul, int man_add,
                                int exp_add, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_bmm_cuda(a, b, c, M, N, K, man_mul, exp_mul, man_add,
                                      exp_add, saturate);
      return;
}

void superfp_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c, int M, int N,
                                   int K, int man_fma, int exp_fma, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma, saturate);
      return;
}

void superfp_quantize_nearest_bmm_fma(Tensor a, Tensor b, Tensor c, int M, int N,
                                    int K, int man_fma, int exp_fma, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_bmm_fma_cuda(a, b, c, M, N, K, 
                                    man_fma, exp_fma, saturate);
      return;
}

void float_quantize_stochastic_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                                  int K, int man_mul, int exp_mul, int man_add,
                                  int exp_add, bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_mm_cuda(a, b, c, M, N, K, man_mul, exp_mul, man_add,
                                        exp_add, subnormals, saturate);
      return;
}

void float_quantize_stochastic_bmm(Tensor a, Tensor b, Tensor c, int M, int N,
                                   int K, int man_mul, int exp_mul, int man_add,
                                   int exp_add, bool subnormals,
                                   bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_bmm_cuda(a, b, c, M, N, K, man_mul, exp_mul,
                                         man_add, exp_add, subnormals, saturate);
      return;
}

void float_quantize_stochastic_mm_fma(Tensor a, Tensor b, Tensor c, int M,
                                      int N, int K, int man_fma, int exp_fma,
                                      bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_mm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                            subnormals, saturate);
      return;
}

void float_quantize_stochastic_bmm_fma(Tensor a, Tensor b, Tensor c, int M,
                                       int N, int K, int man_fma, int exp_fma,
                                       bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_bmm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                             subnormals, saturate);
      return;
}

void fixed_point_quantize_nearest_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                                     int K, int wl_add, int fl_add, int wl_mul,
                                     int fl_mul, bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_nearest_mm_cuda(a, b, c, M, N, K, wl_add, fl_add, wl_mul,
                                           fl_mul, symmetric);
      return;
}

void fixed_point_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c, int M,
                                         int N, int K, int wl_fma, int fl_fma,
                                         bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                               symmetric);
      return;
}

void fixed_point_quantize_stochastic_mm(Tensor a, Tensor b, Tensor c, int M,
                                        int N, int K, int wl_add, int fl_add,
                                        int wl_mul, int fl_mul,
                                        bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_stochastic_mm_cuda(a, b, c, M, N, K, wl_add, fl_add,
                                              wl_mul, fl_mul, symmetric);
      return;
}

void fixed_point_quantize_stochastic_mm_fma(Tensor a, Tensor b, Tensor c, int M,
                                            int N, int K, int wl_fma,
                                            int fl_fma, bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_stochastic_mm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                                  symmetric);
      return;
}

void floating_point_mm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                              CUBLASMatrixType AB_type, CUBLASMatrixType C_type,
                              CUBLASComputeType compute_type, bool pedantic)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_mm_cublas(a, b, c, M, N, K, AB_type, C_type, compute_type, pedantic);
      return;
}

void floating_point_bmm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                               CUBLASMatrixType AB_type, CUBLASMatrixType C_type,
                               CUBLASComputeType compute_type, bool pedantic)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_bmm_cublas(a, b, c, M, N, K, AB_type, C_type, compute_type, pedantic);
      return;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
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
      m.def("p3109_quantize_stochastic", &p3109_quantize_stochastic,
            "Low-Bitwidth P3109 Floating-Point Number Stochastic Quantization (CUDA)");
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
      m.def("p3109_quantize_nearest", &p3109_quantize_nearest,
            "Low-Bitwidth P3109 Floating-Point Number Nearest Quantization (CUDA)");
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


      py::enum_<CUBLASMatrixType>(m, "CUBLASMatrixType", py::arithmetic())
            .value("F32", CUBLASMatrixType::kF32)
            .value("F16", CUBLASMatrixType::kF16)
            .value("BF16", CUBLASMatrixType::kBF16);

      py::enum_<CUBLASComputeType>(m, "CUBLASComputeType", py::arithmetic())
            .value("F32", CUBLASComputeType::kF32)
            .value("F16", CUBLASComputeType::kF16)
            .value("F32_FAST_F16", CUBLASComputeType::kF32FastF16)
            .value("F32_FAST_BF16", CUBLASComputeType::kF32FastBF16)
            .value("F32_FAST_TF32", CUBLASComputeType::kF32FastTF32);

      m.def("create_cublas_handle", &create_cublas_handle, "Creates a new cuBLAS handle");
      m.def("delete_cublas_handle", &delete_cublas_handle, "Deletes the current cuBLAS handle");
      m.def("floating_point_mm_cublas",
            &floating_point_mm_cublas,
            "cuBLAS accelerated matrix multiply, using the specified precision and "
            "compute mode (CUDA)");
      m.def("floating_point_bmm_cublas",
            &floating_point_bmm_cublas,
            "cuBLAS accelerated batched matrix multiply, using the specified precision "
            "and compute mode (CUDA)");
}
