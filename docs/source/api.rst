.. _api-reference:

API reference
=============

The public surface, by module. Everything documented here is importable
from the two packages ``mptorch`` (formats and enums) and ``mptorch.quant``
(quantizers, matrix products, layers).

Number formats and modes
------------------------

.. automodule:: mptorch.number
   :members: RoundMode, SaturationMode, SubnormalsMode, AccumulateAlgorithm, Number, FloatFormat, BinaryK, SuperFP, FormatRangeWarning
   :undoc-members:
   :member-order: bysource

Elementwise quantization
------------------------

.. autofunction:: mptorch.quant.binaryK_quantize

.. autofunction:: mptorch.quant.superfp_quantize

.. autofunction:: mptorch.quant.binaryK_quantize_

.. autofunction:: mptorch.quant.superfp_quantize_

.. autoclass:: mptorch.quant.Quant
   :members: __call__

.. autoclass:: mptorch.quant.Quantizer
   :members: forward

Dot-product arithmetic
----------------------

.. automodule:: mptorch.quant.mac
   :members: SplitMac, FusedMac, Palette
   :no-undoc-members:

Matrix products
---------------

.. automodule:: mptorch.quant.matmul
   :members: qmatmul, qmm, qbmm, as_matmul_formats

.. autoclass:: mptorch.quant.QMatmulFormats

.. autoclass:: mptorch.quant.QMatmul
   :members: forward

.. autofunction:: mptorch.quant.matmul_formats

The schema tier
~~~~~~~~~~~~~~~

One function per kernel, with every schema argument spelled out and no
autograd. See :doc:`gemm` for when to use these instead of ``qmatmul``.

.. autofunction:: mptorch.quant.binaryK_matmul

.. autofunction:: mptorch.quant.superfp_matmul

.. autofunction:: mptorch.quant.binaryK_matmul_fma

.. autofunction:: mptorch.quant.superfp_matmul_fma

.. autofunction:: mptorch.quant.binaryK_matmul_mixed

.. autofunction:: mptorch.quant.superfp_matmul_mixed

.. autofunction:: mptorch.quant.binaryK_matmul_fma_mixed

.. autofunction:: mptorch.quant.superfp_matmul_fma_mixed

Layers
------

.. autoclass:: mptorch.quant.QAffineFormats

.. autoclass:: mptorch.quant.QLinear
   :members: forward

.. autoclass:: mptorch.quant.QConv1d

.. autoclass:: mptorch.quant.QConv2d

.. autoclass:: mptorch.quant.QConv3d

Layer arithmetic factories
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mptorch.quant.binaryK_gemm_formats

.. autofunction:: mptorch.quant.binaryK_gemm_formats_fma

.. autofunction:: mptorch.quant.superfp_gemm_formats

.. autofunction:: mptorch.quant.superfp_gemm_formats_fma
