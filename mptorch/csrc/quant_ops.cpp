// The mptorch op library: the Python module entry point and the schema of
// every op, declared once here and implemented per backend elsewhere.

#include "quant_ops.h"
#include <Python.h>
#include <torch/library.h>

// The extension is loaded as the Python module mptorch._C, so it needs a
// module initializer, but the module itself is empty: the ops are reached
// through torch.ops.mptorch.<op>, which the TORCH_LIBRARY block below
// registers with the dispatcher when the shared object is loaded. Importing
// mptorch._C is what triggers that load.
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

// The schema of each op, in TorchScript schema syntax. The dispatcher
// parses these to type-check calls from Python, and each backend then binds
// an implementation with TORCH_LIBRARY_IMPL under its own key
// (cpu/cpu_ops.cpp, cuda/cuda_ops.cu, autograd_ops.cpp). The mode arguments
// are int, not string: they carry the RoundMode/SaturationMode/
// SubnormalsMode enum values of common/modes.h, which mptorch.number mirrors
// one to one on the Python side. The *_mixed ops take int[] lists, one
// entry per palette slot, and a prec_idx tensor of slot indices per output
// element.
//
// The two ops whose name ends in an underscore write their argument, and say
// so the way ATen's own in-place ops do: `Tensor(a!)` on the argument and on
// the return, which is the same tensor. The annotation is what lets the
// dispatcher's ADInplaceOrView kernel (autograd_ops.cpp) bump the tensor's
// version counter, and what torch.compile's functionalization reads.
TORCH_LIBRARY(mptorch, m)
{
  m.def("binaryK_quant(Tensor a, int K, int P, "
        "int bias, int prng_bits, bool is_signed, "
        "int round_mode, int saturation_mode, int subnormals_mode) -> Tensor");
  m.def("superfp_quant(Tensor a, int man_bits, int exp_bits, int normal_binades, "
        "int bias, int prng_bits, bool is_signed, "
        "int round_mode, int saturation_mode) -> Tensor");
  m.def("binaryK_quant_(Tensor(a!) a, int K, int P, "
        "int bias, int prng_bits, bool is_signed, "
        "int round_mode, int saturation_mode, int subnormals_mode) -> Tensor(a!)");
  m.def("superfp_quant_(Tensor(a!) a, int man_bits, int exp_bits, int normal_binades, "
        "int bias, int prng_bits, bool is_signed, "
        "int round_mode, int saturation_mode) -> Tensor(a!)");
  m.def("narrow_float64(Tensor a, ScalarType dtype) -> Tensor");
  m.def("custom_matmul_binaryK(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "int mul_K, int mul_P, int mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int acc_K, int acc_P, int acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int mul_subnormals_mode, "
        "int acc_saturation_mode, int acc_subnormals_mode, "
        "int mul_prng_bits, int acc_prng_bits) -> Tensor");
  m.def("custom_matmul_superfp(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "int mul_man_bits, int mul_exp_bits, int mul_normal_binades, int mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int acc_man_bits, int acc_exp_bits, int acc_normal_binades, "
        "int acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int acc_saturation_mode, "
        "int mul_prng_bits, int acc_prng_bits) -> Tensor");
  m.def("custom_matmul_binaryK_fma(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "bool fma_quant, int fma_K, int fma_P, int fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int fma_saturation_mode, "
        "int fma_subnormals_mode, int fma_prng_bits) -> Tensor");
  m.def("custom_matmul_superfp_fma(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "bool fma_quant, int fma_man_bits, int fma_exp_bits, int fma_normal_binades, "
        "int fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int fma_saturation_mode, "
        "int fma_prng_bits) -> Tensor");
  m.def("custom_matmul_binaryK_mixed(Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b, "
        "int[] mul_K, int[] mul_P, int[] mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int[] acc_K, int[] acc_P, int[] acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int mul_subnormals_mode, "
        "int acc_saturation_mode, int acc_subnormals_mode, "
        "int mul_prng_bits, int acc_prng_bits) -> Tensor");
  m.def("custom_matmul_superfp_mixed(Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b, "
        "int[] mul_man_bits, int[] mul_exp_bits, int[] mul_normal_binades, int[] mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int[] acc_man_bits, int[] acc_exp_bits, int[] acc_normal_binades, "
        "int[] acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int acc_saturation_mode, "
        "int mul_prng_bits, int acc_prng_bits) -> Tensor");
  m.def("custom_matmul_binaryK_fma_mixed(Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b, "
        "bool fma_quant, int[] fma_K, int[] fma_P, int[] fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int fma_saturation_mode, "
        "int fma_subnormals_mode, int fma_prng_bits) -> Tensor");
  m.def("custom_matmul_superfp_fma_mixed(Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b, "
        "bool fma_quant, int[] fma_man_bits, int[] fma_exp_bits, int[] fma_normal_binades, "
        "int[] fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int fma_saturation_mode, "
        "int fma_prng_bits) -> Tensor");
}
