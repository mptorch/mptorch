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
//
// The four *_accumulated ops are the single-format GEMMs under an accumulate
// algorithm other than NAIVE (common/gemm_accumulate.h): their twin's
// arguments, then a block size and an outer format, spelled as the family
// spells a format. They are ops of their own, rather than nine more arguments
// on the four they extend, because an argument costs every call that does not
// use it: with them appended and defaulted a NAIVE call measured 1.0-1.4 us
// slower through torch.ops, a tenth of the whole call (dev/gemm_roadmap.md,
// R-2). The four *_mixed ops are NAIVE only.
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
  m.def("custom_matmul_binaryK_accumulated(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "int mul_K, int mul_P, int mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int acc_K, int acc_P, int acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int mul_subnormals_mode, "
        "int acc_saturation_mode, int acc_subnormals_mode, "
        "int mul_prng_bits, int acc_prng_bits, "
        "int block_size, bool outer_quant, int outer_K, int outer_P, int outer_bias, "
        "bool outer_is_signed, int outer_saturation_mode, int outer_subnormals_mode, "
        "int outer_prng_bits) -> Tensor");
  m.def("custom_matmul_superfp_accumulated(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "int mul_man_bits, int mul_exp_bits, int mul_normal_binades, int mul_bias, bool mul_is_signed, "
        "bool accumulate_quant, int acc_man_bits, int acc_exp_bits, int acc_normal_binades, "
        "int acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int acc_saturation_mode, "
        "int mul_prng_bits, int acc_prng_bits, "
        "int block_size, bool outer_quant, int outer_man_bits, int outer_exp_bits, "
        "int outer_normal_binades, int outer_bias, bool outer_is_signed, "
        "int outer_saturation_mode, int outer_prng_bits) -> Tensor");
  m.def("custom_matmul_binaryK_fma_accumulated(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "bool fma_quant, int fma_K, int fma_P, int fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int fma_saturation_mode, "
        "int fma_subnormals_mode, int fma_prng_bits, "
        "int block_size, bool outer_quant, int outer_K, int outer_P, int outer_bias, "
        "bool outer_is_signed, int outer_saturation_mode, int outer_subnormals_mode, "
        "int outer_prng_bits) -> Tensor");
  m.def("custom_matmul_superfp_fma_accumulated(Tensor a, Tensor b, bool trans_a, bool trans_b, "
        "bool fma_quant, int fma_man_bits, int fma_exp_bits, int fma_normal_binades, "
        "int fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int fma_saturation_mode, "
        "int fma_prng_bits, "
        "int block_size, bool outer_quant, int outer_man_bits, int outer_exp_bits, "
        "int outer_normal_binades, int outer_bias, bool outer_is_signed, "
        "int outer_saturation_mode, int outer_prng_bits) -> Tensor");
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
  // The conv ops (common/gemm_gather.h, common/gemm_conv_host.h): one pass of
  // a convolution, forward or either gradient, as the GEMM of its twin with
  // the operands gathered in place. A single-format conv op is its
  // *_accumulated twin's schema, so it takes every AccumulateAlgorithm, and a
  // palette conv op its *_mixed twin's; the operand pair is followed by the
  // geometry instead of the transpose flags.
  m.def("custom_conv_binaryK(Tensor a, Tensor b, int conv_pass, int[] out_size, int[] stride, "
        "int[] padding, int[] dilation, int groups, int mul_K, int mul_P, int mul_bias, "
        "bool mul_is_signed, bool accumulate_quant, int acc_K, int acc_P, int acc_bias, "
        "bool acc_is_signed, int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int mul_subnormals_mode, int acc_saturation_mode, "
        "int acc_subnormals_mode, int mul_prng_bits, int acc_prng_bits, int block_size, "
        "bool outer_quant, int outer_K, int outer_P, int outer_bias, bool outer_is_signed, "
        "int outer_saturation_mode, int outer_subnormals_mode, int outer_prng_bits) -> Tensor");
  m.def("custom_conv_binaryK_mixed(Tensor a, Tensor b, Tensor prec_idx, int conv_pass, "
        "int[] out_size, int[] stride, int[] padding, int[] dilation, int groups, "
        "int[] mul_K, int[] mul_P, int[] mul_bias, bool mul_is_signed, bool accumulate_quant, "
        "int[] acc_K, int[] acc_P, int[] acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, int mul_saturation_mode, "
        "int mul_subnormals_mode, int acc_saturation_mode, int acc_subnormals_mode, "
        "int mul_prng_bits, int acc_prng_bits) -> Tensor");
  m.def("custom_conv_binaryK_fma(Tensor a, Tensor b, int conv_pass, int[] out_size, "
        "int[] stride, int[] padding, int[] dilation, int groups, bool fma_quant, int fma_K, "
        "int fma_P, int fma_bias, bool fma_is_signed, int accumulate_algorithm, "
        "int round_mode, int fma_saturation_mode, int fma_subnormals_mode, int fma_prng_bits, "
        "int block_size, bool outer_quant, int outer_K, int outer_P, int outer_bias, "
        "bool outer_is_signed, int outer_saturation_mode, int outer_subnormals_mode, "
        "int outer_prng_bits) -> Tensor");
  m.def("custom_conv_binaryK_fma_mixed(Tensor a, Tensor b, Tensor prec_idx, int conv_pass, "
        "int[] out_size, int[] stride, int[] padding, int[] dilation, int groups, "
        "bool fma_quant, int[] fma_K, int[] fma_P, int[] fma_bias, bool fma_is_signed, "
        "int accumulate_algorithm, int round_mode, int fma_saturation_mode, "
        "int fma_subnormals_mode, int fma_prng_bits) -> Tensor");
  m.def("custom_conv_superfp(Tensor a, Tensor b, int conv_pass, int[] out_size, int[] stride, "
        "int[] padding, int[] dilation, int groups, int mul_man_bits, int mul_exp_bits, "
        "int mul_normal_binades, int mul_bias, bool mul_is_signed, bool accumulate_quant, "
        "int acc_man_bits, int acc_exp_bits, int acc_normal_binades, int acc_bias, "
        "bool acc_is_signed, int accumulate_algorithm, int round_mode, "
        "int mul_saturation_mode, int acc_saturation_mode, int mul_prng_bits, "
        "int acc_prng_bits, int block_size, bool outer_quant, int outer_man_bits, "
        "int outer_exp_bits, int outer_normal_binades, int outer_bias, bool outer_is_signed, "
        "int outer_saturation_mode, int outer_prng_bits) -> Tensor");
  m.def("custom_conv_superfp_mixed(Tensor a, Tensor b, Tensor prec_idx, int conv_pass, "
        "int[] out_size, int[] stride, int[] padding, int[] dilation, int groups, "
        "int[] mul_man_bits, int[] mul_exp_bits, int[] mul_normal_binades, int[] mul_bias, "
        "bool mul_is_signed, bool accumulate_quant, int[] acc_man_bits, int[] acc_exp_bits, "
        "int[] acc_normal_binades, int[] acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, int mul_saturation_mode, "
        "int acc_saturation_mode, int mul_prng_bits, int acc_prng_bits) -> Tensor");
  m.def("custom_conv_superfp_fma(Tensor a, Tensor b, int conv_pass, int[] out_size, "
        "int[] stride, int[] padding, int[] dilation, int groups, bool fma_quant, "
        "int fma_man_bits, int fma_exp_bits, int fma_normal_binades, int fma_bias, "
        "bool fma_is_signed, int accumulate_algorithm, int round_mode, "
        "int fma_saturation_mode, int fma_prng_bits, int block_size, bool outer_quant, "
        "int outer_man_bits, int outer_exp_bits, int outer_normal_binades, int outer_bias, "
        "bool outer_is_signed, int outer_saturation_mode, int outer_prng_bits) -> Tensor");
  m.def("custom_conv_superfp_fma_mixed(Tensor a, Tensor b, Tensor prec_idx, int conv_pass, "
        "int[] out_size, int[] stride, int[] padding, int[] dilation, int groups, "
        "bool fma_quant, int[] fma_man_bits, int[] fma_exp_bits, int[] fma_normal_binades, "
        "int[] fma_bias, bool fma_is_signed, int accumulate_algorithm, int round_mode, "
        "int fma_saturation_mode, int fma_prng_bits) -> Tensor");
  // The block-format ops (common/block_decode.h; mptorch.number.BlockFormat).
  // `fmt` is the format's descriptor, an int list in BlockFmtField's order,
  // and the three floats are its largest element, its largest scale and the
  // per-tensor scale (1 for a format without one). block_pack writes uint8
  // codes and scales; block_unpack reads them back as `dtype`, of whose
  // packed axis `cols` is the unpadded length; block_quant is the two fused,
  // with no packed intermediate. custom_matmul_block multiplies two packed
  // operands, A [M, K] and B [N, K] (or stored with K as their rows), in the
  // binaryK accumulate format given, under any AccumulateAlgorithm: the
  // accumulation tail is the *_accumulated ops'.
  m.def("block_pack(Tensor x, int[] fmt, float elem_max, float scale_max, float tensor_scale, "
        "int round_mode) -> (Tensor, Tensor)");
  m.def("block_unpack(Tensor data, Tensor scales, int cols, int[] fmt, float elem_max, "
        "float scale_max, float tensor_scale, ScalarType dtype) -> Tensor");
  m.def("block_quant(Tensor x, int[] fmt, float elem_max, float scale_max, float tensor_scale, "
        "int round_mode) -> Tensor");
  m.def("block_quant_(Tensor(a!) x, int[] fmt, float elem_max, float scale_max, "
        "float tensor_scale, int round_mode) -> Tensor(a!)");
  m.def("custom_matmul_block(Tensor a_data, Tensor a_scales, int a_cols, bool trans_a, "
        "Tensor b_data, Tensor b_scales, int b_cols, bool trans_b, "
        "int[] a_fmt, float a_elem_max, float a_scale_max, float a_tensor_scale, "
        "int[] b_fmt, float b_elem_max, float b_scale_max, float b_tensor_scale, "
        "bool fused, bool accumulate_quant, int acc_K, int acc_P, int acc_bias, bool acc_is_signed, "
        "int accumulate_algorithm, int round_mode, int acc_saturation_mode, "
        "int acc_subnormals_mode, int acc_prng_bits, "
        "int block_size, bool outer_quant, int outer_K, int outer_P, int outer_bias, "
        "bool outer_is_signed, int outer_saturation_mode, int outer_subnormals_mode, "
        "int outer_prng_bits) -> Tensor");
}
