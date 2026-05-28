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
}
