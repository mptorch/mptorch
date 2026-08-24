#include "quant_ops.h"
#include <Python.h>
#include <torch/library.h>

extern "C"
{
  PyObject *PyInit__C(void)
  {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
  }
}

TORCH_LIBRARY(mptorch, m)
{
  m.def("binaryK_quant(Tensor a, int K, int P, "
        "int bias, int prng_bits, bool is_signed, "
        "int round_mode, int saturation_mode, int subnormals_mode) -> Tensor");
  m.def("superfp_quant(Tensor a, int man_bits, int exp_bits, int normal_binades, "
        "int bias, int prng_bits, bool is_signed, "
        "int round_mode, int saturation_mode) -> Tensor");
  m.def("custom_matmul_binaryK(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "int mul_K, int mul_P, int mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int acc_K, int acc_P, int acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, int saturation_mode, "
        "int subnormals_mode) -> Tensor");
  m.def("custom_matmul_superfp(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "int mul_man_bits, int mul_exp_bits, int mul_normal_binades, int mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int acc_man_bits, int acc_exp_bits, int acc_normal_binades, "
        "int acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, int saturation_mode) -> Tensor");
  m.def("custom_matmul_binaryK_fma(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "bool fma_quant, int fma_K, int fma_P, int fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int saturation_mode, "
        "int subnormals_mode) -> Tensor");
  m.def("custom_matmul_superfp_fma(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "bool fma_quant, int fma_man_bits, int fma_exp_bits, int fma_normal_binades, "
        "int fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int saturation_mode) -> Tensor");
}
