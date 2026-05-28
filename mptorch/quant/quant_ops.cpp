#include "quant_ops.h"
#include <torch/library.h>

TORCH_LIBRARY(mptorch, m)
{
    m.def(
        "binaryK_quant(Tensor a, int K, int P, "
        "int bias, int prng_bits, bool is_signed, "
        "int round_mode, int saturation_mode, int subnormals_mode) -> Tensor)");
}