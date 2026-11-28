"""The factories that fill a layer's format container with quantized GEMMs.

Each factory returns the container a layer reads its arithmetic from (a
``QAffineFormats`` for ``QLinear`` and the convolutions, a ``QMatmulFormats``
for ``QMatmul``), its math hooks calling the kernel ops of
:mod:`mptorch.quant.ops`. They are named after the operation whose layer they
configure, as :mod:`mptorch.quant.modules` and :mod:`mptorch.quant.functional`
name theirs. Every public name is importable from :mod:`mptorch.quant`, and
this package re-exports none.
"""

# One module per operation:
#
#   linear     the *_gemm_formats factories: binaryK, superfp, cross-family, fixed point, block
#   matmul     matmul_formats, block_matmul_formats
#   conv       conv_formats, and the plain convolution a layer runs without a hook
#   _hooks     the hook builders the factories and mptorch.quant.GemmFormats compose: one per
#              pass and per kind of GEMM, and the containers' call contract
#   _exact     fixed-point GEMMs whose every step is exact, run on a vendor GEMM
