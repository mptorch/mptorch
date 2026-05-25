#include "../mm_ops.h"
#include <torch/library.h>

TORCH_LIBRARY(mptorch, m)
{
  m.def(
      "fp_mm(Tensor out, Tensor a, Tensor b, int man_add, int exp_add, int "
      "man_mul, int exp_mul, int man_fma, int exp_fma, int round_mode, int "
      "subnormals_mode, bool saturate, bool compensated, bool use_fma, int "
      "rbits_add, int rbits_mul, int rbits_fma) -> ()");
  m.def(
      "superfp_mm(Tensor out, Tensor a, Tensor b, int man_add, int exp_add, int "
      "man_mul, int exp_mul, int man_fma, int exp_fma, int binades_add_l, int "
      "binades_add_u, int binades_mul_l, int binades_mul_u, int binades_fma_l, "
      "int binades_fma_u, bool saturate, bool use_fma) -> ()");
  m.def(
      "fxp_mm(Tensor out, Tensor a, Tensor b, int wl_add, int fl_add, int "
      "wl_mul, int fl_mul, int wl_fma, int fl_fma, int round_mode, bool "
      "symmetric, bool use_fma) -> ()");
}
