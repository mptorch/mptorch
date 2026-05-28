#include "../quant_ops.h"
#include <torch/library.h>

TORCH_LIBRARY_IMPL(mptorch, CPU, m)
{
    m.def("binaryK_quant", TORCH_FN(binaryK_quantize_cpu));
}